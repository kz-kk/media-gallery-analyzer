#!/usr/bin/env python3
# coding: utf-8
"""
audio_analyzer_v2.py
- librosaで音響特徴量を抽出
- LM Studioでテキスト分析してタグ生成
- Whisperで歌詞文字起こし
- Meilisearch/Qdrantに保存
"""
import librosa
import ffmpeg
import whisper
import numpy as np
import json
import time
import os
from pathlib import Path
import logging
import warnings
from typing import Dict, List, Optional, Tuple
from secure_logging import safe_debug_print
import threading

logging.basicConfig(level=logging.INFO)

# Suppress noisy warnings for known fallbacks (m4a via audioread, librosa kw rename)
try:
    warnings.filterwarnings(
        "ignore",
        message=r"get_duration\(\) keyword argument 'filename' has been renamed to 'path'",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"PySoundFile failed. Trying audioread instead.",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"librosa\.core\.audio\.__audioread_load",
        category=FutureWarning,
    )
except Exception:
    pass

def _load_repr_segment(audio_path: Path, target_sr: int = 16000):
    """代表区間をロード（冒頭静音回避）"""
    try:
        y, sr = _safe_load_audio(str(audio_path), target_sr=target_sr)
        import librosa as _lib
        duration = _lib.get_duration(y=y, sr=sr)
        if duration > 180:
            start = int(sr * 60)
            end = int(sr * 180)
            return y[start:end], sr
        elif duration > 90:
            start = int(sr * 30)
            end = int(sr * 90)
            return y[start:end], sr
        else:
            return y, sr
    except Exception as e:
        safe_debug_print(f"repr segment load error: {e}")
        return None, None

def _probe_audio_sr(audio_path: str) -> Optional[int]:
    try:
        info = ffmpeg.probe(audio_path)
        for s in info.get('streams', []):
            if s.get('codec_type') == 'audio':
                sr = s.get('sample_rate')
                if sr:
                    return int(sr)
    except Exception:
        return None
    return None

def _decode_with_ffmpeg(audio_path: str, target_sr: Optional[int] = None,
                        offset: Optional[float] = None, duration: Optional[float] = None,
                        mono: bool = True):
    """ffmpegで任意の区間をPCM float32でデコード"""
    try:
        sr = int(target_sr) if target_sr else (_probe_audio_sr(audio_path) or 16000)
        i = ffmpeg.input(audio_path, ss=offset if offset else None, t=duration if duration else None)
        out = i.output('pipe:', format='f32le', ac=(1 if mono else 2), ar=sr)
        out = out.run(capture_stdout=True, capture_stderr=True, quiet=True)
        import numpy as _np
        y = _np.frombuffer(out[0], dtype=_np.float32)
        if mono:
            return y, sr
        # ステレオのときは2chに整形
        if y.size % 2 == 0:
            y = y.reshape(-1, 2).mean(axis=1)
        return y.astype(_np.float32), sr
    except Exception as e:
        safe_debug_print(f"ffmpeg decode failed: {e}")
        return None, None

def _safe_load_audio(audio_path: str, target_sr: Optional[int] = None,
                     offset: Optional[float] = None, duration: Optional[float] = None):
    """soundfile/librosaで失敗時にffmpegでデコードして安全に読み込む"""
    try:
        import librosa as _lib
        y, sr = _lib.load(audio_path, sr=target_sr, mono=True, offset=offset or 0.0,
                          duration=duration if (duration and duration > 0) else None)
        return y, sr
    except Exception:
        return _decode_with_ffmpeg(audio_path, target_sr=target_sr, offset=offset, duration=duration, mono=True)

def _detect_melody_with_crepe(y, sr):
    """CREPEでメロディ検出（任意）"""
    try:
        import crepe
        if y is None or sr is None:
            return {}
        # CREPEは16k/monoでOK
        time, frequency, confidence, activation = crepe.predict(y, sr, viterbi=True, step_size=20)
        import numpy as _np
        conf = _np.array(confidence, dtype=float)
        freq = _np.array(frequency, dtype=float)
        voiced_mask = (conf >= 0.5) & (freq > 0)
        vr = float(_np.mean(voiced_mask)) if voiced_mask.size else 0.0
        # 最長連続
        longest = 0
        cur = 0
        for v in voiced_mask.astype(int):
            if v:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 0
        # step_size=20ms
        longest_sec = longest * 0.02
        f0_med = float(_np.median(freq[voiced_mask])) if _np.any(voiced_mask) else 0.0
        return {
            'crepe_voiced_ratio': float(vr),
            'crepe_longest_run_sec': float(longest_sec),
            'crepe_f0_median': float(f0_med),
            'crepe_has_melody': bool(vr >= 0.08 or longest_sec >= 0.8)
        }
    except Exception as e:
        safe_debug_print(f"CREPE not available or failed: {e}")
        return {}

def _infer_instruments_with_panns(audio_path: Path):
    """PANNs(AudioSet)で楽器/シーンを推定（任意）"""
    try:
        from panns_inference import AudioTagging
        import numpy as _np
        import librosa as _lib
        # 代表区間をロード（32kにリサンプル）
        y, sr = _load_repr_segment(audio_path, target_sr=32000)
        if y is None:
            return []
        at = AudioTagging(checkpoint_path=None, device='cpu')
        # panns_inference.AudioTagging.inference expects audio array only
        (clipwise_output, _embedding) = at.inference(_np.array(y))
        labels = getattr(at, 'labels', [])
        scores = clipwise_output[0]
        label_scores = list(zip(labels, scores))
        label_scores.sort(key=lambda x: x[1], reverse=True)
        tags = []
        for lab, sc in label_scores[:20]:
            if sc < 0.25:  # しきい値
                continue
            l = lab.lower()
            if any(k in l for k in ['violin']):
                tags += ['ヴァイオリン', 'ストリングス']
            if any(k in l for k in ['cello']):
                tags += ['チェロ', 'ストリングス']
            if any(k in l for k in ['viola']):
                tags += ['ヴィオラ', 'ストリングス']
            if any(k in l for k in ['string', 'orchestra', 'orchestral']):
                tags += ['ストリングス', 'オーケストラ']
            if any(k in l for k in ['flute', 'clarinet', 'oboe', 'bassoon', 'woodwind']):
                tags += ['木管']
            if any(k in l for k in ['trumpet', 'trombone', 'horn', 'tuba', 'brass']):
                tags += ['金管']
            if any(k in l for k in ['piano']):
                tags += ['ピアノ']
            if any(k in l for k in ['organ']):
                tags += ['オルガン']
            if any(k in l for k in ['timpani', 'snare', 'cymbal', 'drum', 'percussion']):
                tags += ['打楽器', 'ドラム']
        # ユニーク
        uniq = []
        seen = set()
        for t in tags:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        return uniq
    except Exception as e:
        safe_debug_print(f"PANNs not available or failed: {e}")
        return []

def extract_audio_features(audio_path: Path) -> Dict:
    """librosaで音響特徴量を抽出"""
    try:
        safe_debug_print(f"音声ファイルを読み込み中: {audio_path}")
        start_time = time.time()
        # 音声を読み込み
        y, sr = librosa.load(str(audio_path), sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        safe_debug_print(f"音声読み込み完了: {time.time() - start_time:.1f}秒, 長さ: {duration:.1f}秒")
        
        # 基本的な特徴量
        features = {
            "duration": float(duration),
            "sample_rate": int(sr),
        }
        
        # 代表区間の抽出（冒頭が静かなクラシック対策）
        if duration > 180:
            start = int(sr * 60)
            end = int(sr * 180)
            y_trimmed = y[start:end]
        elif duration > 90:
            start = int(sr * 30)
            end = int(sr * 90)
            y_trimmed = y[start:end]
        else:
            y_trimmed = y
            
        # リズム特徴量
        tempo, beats = librosa.beat.beat_track(y=y_trimmed, sr=sr)
        features["tempo"] = float(tempo)
        features["beat_count"] = len(beats)
        # 倍テンポ誤検出の簡易補正（長尺・低ZCRの器楽で発生しやすい）
        try:
            window_dur = float(len(y_trimmed)) / float(sr)
            est_bpm = (len(beats) * 60.0 / window_dur) if window_dur > 0 else 0.0
        except Exception:
            est_bpm = 0.0
        tempo_refined = float(tempo)
        if tempo_refined >= 140.0 and est_bpm > 0 and tempo_refined > est_bpm * 1.35:
            tempo_refined = tempo_refined / 2.0
        # 全体が長尺かつ持続音中心なら更にハーフテンポを検討
        try:
            full_duration = float(features.get("duration") or duration)
            zcr_val = float(features.get("zero_crossing_rate") or 0.0)
        except Exception:
            full_duration = duration
            zcr_val = 0.0
        if full_duration >= 600 and tempo_refined >= 140.0 and zcr_val <= 0.06:
            tempo_refined = tempo_refined / 2.0
        features["tempo_refined"] = float(tempo_refined)
        
        # スペクトル特徴量
        spectral_centroids = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        features["spectral_centroid_std"] = float(np.std(spectral_centroids))

        # スペクトルコントラスト（明暗/シャープさの指標）
        try:
            spec_contrast = librosa.feature.spectral_contrast(y=y_trimmed, sr=sr)
            features["spectral_contrast_mean"] = float(np.mean(spec_contrast))
        except Exception:
            features["spectral_contrast_mean"] = 0.0
        
        # ゼロクロッシングレート（音の質感）
        zero_crossings = librosa.zero_crossings(y_trimmed, pad=False)
        features["zero_crossing_rate"] = float(sum(zero_crossings)) / len(y_trimmed)
        
        # MFCC（音色特徴）
        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        features["mfcc_mean"] = [float(np.mean(mfcc)) for mfcc in mfccs]
        features["mfcc_std"] = [float(np.std(mfcc)) for mfcc in mfccs]
        
        # RMS（音量）
        rms = librosa.feature.rms(y=y_trimmed)
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))
        
        # 音高検出（ピッチ）: piptrack + pyin の併用とメロディ連続性の簡易推定
        # 1) piptrack（速いがノイジー）
        pitches, magnitudes = librosa.piptrack(y=y_trimmed, sr=sr)
        pitch_values = []
        per_frame_pitch = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = float(pitches[index, t])
            if pitch > 0:
                pitch_values.append(pitch)
                per_frame_pitch.append(pitch)
            else:
                per_frame_pitch.append(0.0)
        # 2) pyin（モノフォニック推定・より頑健）
        f0_pyin = None
        try:
            f0_pyin = librosa.pyin(y_trimmed, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        except Exception:
            f0_pyin = (None, None, None)
        f0_vals = None
        pyin_voiced_ratio = 0.0
        f0_median = 0.0
        f0_iqr = 0.0
        try:
            if isinstance(f0_pyin, tuple):
                f0_series = f0_pyin[0]
            else:
                f0_series = f0_pyin
            if f0_series is not None:
                import numpy as _np
                f0_arr = _np.array(f0_series, dtype=float)
                voiced = f0_arr[_np.isfinite(f0_arr)]
                total = len(f0_arr)
                pyin_voiced_ratio = float(len(voiced)) / float(total or 1)
                if voiced.size > 0:
                    f0_median = float(_np.median(voiced))
                    q75, q25 = _np.percentile(voiced, [75 ,25])
                    f0_iqr = float(q75 - q25)
                # 最長連続有声
                is_voiced = _np.isfinite(f0_arr).astype(int)
                longest_run_pyin = 0
                cur = 0
                for v in is_voiced:
                    if v:
                        cur += 1
                        if cur > longest_run_pyin:
                            longest_run_pyin = cur
                    else:
                        cur = 0
                hop_length = 512
                frame_duration = hop_length / float(sr)
                longest_run_seconds_pyin = float(longest_run_pyin) * frame_duration
                features["melody_longest_run_sec_pyin"] = longest_run_seconds_pyin
        except Exception:
            pass
        
        if pitch_values:
            features["pitch_mean"] = float(np.mean(pitch_values))
            features["pitch_std"] = float(np.std(pitch_values))
        else:
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0
        features["pyin_voiced_ratio"] = float(pyin_voiced_ratio)
        features["f0_median"] = float(f0_median)
        features["f0_iqr"] = float(f0_iqr)

        # フレームごとの有声（ピッチ>0）割合と最長連続秒数
        try:
            hop_length = 512
            frame_duration = hop_length / float(sr)
        except Exception:
            frame_duration = 0.023
        voiced_frames = [1 if p > 0 else 0 for p in per_frame_pitch]
        voiced_ratio = float(sum(voiced_frames)) / float(len(voiced_frames) or 1)
        # 最長連続
        longest_run = 0
        cur = 0
        for v in voiced_frames:
            if v:
                cur += 1
                longest_run = max(longest_run, cur)
            else:
                cur = 0
        longest_run_seconds = float(longest_run) * frame_duration
        features["voiced_frame_ratio"] = voiced_ratio
        features["melody_longest_run_sec"] = longest_run_seconds
            
        # スペクトルロールオフ（明るさの指標）
        rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)
        features["spectral_rolloff_mean"] = float(np.mean(rolloff))

        # 明暗スコア: 低域/高域のエネルギー比 + セントロイドを総合
        S, phase = librosa.magphase(librosa.stft(y_trimmed, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        # 1500Hzを境に低域/高域に分ける
        split_idx = int(np.argmax(freqs > 1500)) or len(freqs) - 1
        low_energy = float(np.sum(S[:split_idx, :]))
        high_energy = float(np.sum(S[split_idx:, :]))
        low_high_ratio = (low_energy + 1e-9) / (high_energy + 1e-9)
        features["low_high_ratio"] = low_high_ratio

        # brightness_score: 0(暗)〜1(明) に正規化
        # セントロイド(0-8000目安)と高域比率から簡易スコア
        centroid_norm = min(max(features["spectral_centroid_mean"] / 8000.0, 0.0), 1.0)
        high_ratio = 1.0 / (1.0 + low_high_ratio)  # 高域優勢なら1に近い
        brightness_score = float(np.clip(0.6 * centroid_norm + 0.4 * high_ratio, 0.0, 1.0))
        features["brightness_score"] = brightness_score

        # 調性推定（キー/モード）: Krumhansl-Schmuckler法に類似の簡易法
        try:
            chroma = librosa.feature.chroma_cqt(y=y_trimmed, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            # トーナル指標: 各フレームの最大値の平均（ピークの強さ）
            chroma_max = np.max(chroma, axis=0)
            tonal_presence = float(np.mean(chroma_max))
            # ピーキーさ（低エントロピーほどピーキー）
            eps = 1e-9
            probs = chroma / (np.sum(chroma, axis=0, keepdims=True) + eps)
            ent = -np.sum(probs * np.log(probs + eps), axis=0) / np.log(12.0)
            chroma_peakiness = float(1.0 - np.mean(ent))
            features["tonal_presence"] = tonal_presence
            features["chroma_peakiness"] = chroma_peakiness
            # メジャー/マイナーのテンプレート（簡易）
            maj_template = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
            min_template = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
            # 各12キーにローテートして相関を評価
            maj_scores = []
            min_scores = []
            for i in range(12):
                maj_scores.append(np.corrcoef(chroma_mean, np.roll(maj_template, i))[0,1])
                min_scores.append(np.corrcoef(chroma_mean, np.roll(min_template, i))[0,1])
            maj_scores = np.nan_to_num(np.array(maj_scores), nan=0.0)
            min_scores = np.nan_to_num(np.array(min_scores), nan=0.0)
            maj_key = int(np.argmax(maj_scores))
            min_key = int(np.argmax(min_scores))
            maj_conf = float(np.max(maj_scores))
            min_conf = float(np.max(min_scores))
            key_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
            if maj_conf >= min_conf:
                features["key"] = key_names[maj_key]
                features["mode"] = "major"
                features["key_confidence"] = maj_conf
            else:
                features["key"] = key_names[min_key]
                features["mode"] = "minor"
                features["key_confidence"] = min_conf
        except Exception as e:
            safe_debug_print(f"キー推定エラー: {e}")
            features["key"] = ""
            features["mode"] = ""
            features["key_confidence"] = 0.0
        
        # メロディ有無の簡易判定: 有声割合/連続 or トーナル指標が一定以上 かつ 持続音優勢
        try:
            harm_ratio = float(features.get("harmonic_ratio") or 0.0)
        except Exception:
            harm_ratio = 0.0
        tonal_presence = float(features.get("tonal_presence") or 0.0)
        chroma_peakiness = float(features.get("chroma_peakiness") or 0.0)
        # piptrack指標に加え、pyinの有声比や最長連続有声も参照
        pyin_long = float(features.get("melody_longest_run_sec_pyin") or 0.0)
        has_melody = (
            (
                voiced_ratio >= 0.08 or longest_run_seconds >= 0.8 or
                tonal_presence >= 0.22 or chroma_peakiness >= 0.3 or
                pyin_voiced_ratio >= 0.08 or pyin_long >= 0.8
            ) and harm_ratio >= 0.45
        )
        features["has_melody"] = bool(has_melody)

        safe_debug_print(f"抽出した特徴量: {len(features)}個")
        
        return features
        
    except Exception as e:
        safe_debug_print(f"音響特徴量抽出エラー: {e}")
        return {}

def classify_audio_type(features: Dict, transcription: Optional[Dict]) -> str:
    """『music / speech / sfx / ambiguous』を判定。
    - 歌唱の誤検出（speech扱い）を抑えるため、メロディ指標を強く考慮
    - 明度（brightness）条件は音楽判定から除外（暗めの楽曲も許容）
    """
    # 基本量
    tempo = float(features.get("tempo_refined", features.get("tempo") or 0))
    beat_count = int(features.get("beat_count") or 0)
    duration = float(features.get("duration") or 0)
    transcript = (transcription or {}).get("text", "") if isinstance(transcription, dict) else ""
    transcript_len = len(transcript.strip())

    # 音楽性の補助指標
    has_melody = bool(features.get("has_melody"))
    tonal_presence = float(features.get("tonal_presence") or 0)
    chroma_peakiness = float(features.get("chroma_peakiness") or 0)

    # SFX（効果音）強調: 短尺・低ビート・文字起こしほぼ無し・高ZCR/高スペクトルコントラスト
    zcr = float(features.get("zero_crossing_rate") or 0)
    spectral_contrast = float(features.get("spectral_contrast_mean") or 0)
    likely_sfx = (
        duration <= 6.0 and beat_count <= 2 and tempo < 110 and transcript_len < 10
    ) or (
        duration <= 10.0 and beat_count <= 2 and transcript_len == 0 and (zcr >= 0.12 or spectral_contrast >= 15)
    )
    if likely_sfx:
        return "sfx"

    # 会話優先ロジック：声の存在があり、強い音楽証拠が無い場合は会話
    voice_present = bool(features.get("voice_present"))
    perc = float(features.get("percussive_ratio") or 0.0)
    onset_rate = float(features.get("onset_rate") or 0.0)
    strong_music = (
        (has_melody and (tempo >= 100 or beat_count >= 10)) or
        (tempo >= 110 and beat_count >= 12) or
        (tonal_presence >= 0.30 and chroma_peakiness >= 0.35)
    )
    # バラード/低テンポ歌唱の救済: 歌詞が十分に検出されたら音楽扱いへ寄せる
    if has_melody and isinstance(transcription, dict):
        _txt = (transcription.get("text") or "").strip()
        if len(_txt) >= 20:
            strong_music = True
    if voice_present and transcript_len >= 10 and not strong_music:
        return "speech"

    # 楽曲（歌詞あり）: 音楽証拠が十分
    if transcript_len >= 12 and strong_music:
        return "music"

    # 強い会話判定：短〜中尺で文字起こし十分、ビート弱
    if transcript_len >= 25 and duration <= 90 and (tempo < 95) and (beat_count <= 6):
        return "speech"

    # 音楽判定（器楽寄り/歌詞少ない）
    if transcript_len < 10 and tempo >= 90 and beat_count >= 12:
        return "music"

    # フォールバック：文字起こし皆無で短尺かつビート弱→SFX寄り
    if transcript_len == 0 and duration <= 20 and beat_count <= 2 and tempo < 110:
        return "sfx"

    return "ambiguous"

def features_to_description(features: Dict) -> str:
    """音響特徴量を自然言語の説明文に変換"""
    descriptions = []
    mode = features.get("mode", "")
    brightness = features.get("brightness_score", 0.0)

    # テンポの説明（補正後を優先）
    tempo = features.get("tempo_refined", features.get("tempo", 0))
    if tempo > 0:
        tempo_phrase = ""
        if tempo < 60:
            tempo_phrase = "非常にゆったり"
        elif tempo < 90:
            tempo_phrase = "ゆったり"
        elif tempo < 120:
            tempo_phrase = "中程度"
        elif tempo < 140:
            tempo_phrase = "やや速い"
        else:
            tempo_phrase = "速い"
        if mode == "minor" and brightness < 0.45 and tempo >= 120:
            descriptions.append(f"{tempo_phrase}テンポ（約{int(tempo)} BPM）ながら陰りのある雰囲気")
        else:
            descriptions.append(f"{tempo_phrase}テンポ（約{int(tempo)} BPM）")
    
    # 音の明るさ（スペクトルセントロイド）
    # 明暗（明るさ）の説明: brightness_score を優先
    if brightness > 0:
        if brightness < 0.35:
            descriptions.append("暗めの音色")
        elif brightness < 0.6:
            descriptions.append("バランスの取れた音色")
        else:
            # マイナー調かつ高明度の場合は中和表現
            if mode == "minor":
                descriptions.append("やや明るさも感じるが陰影のある音色")
            else:
                descriptions.append("明るい音色")
    
    # 音量の変化（RMS）
    rms_std = features.get("rms_std", 0)
    if rms_std > 0:
        if rms_std < 0.05:
            descriptions.append("音量の変化が少なく安定した演奏")
        elif rms_std < 0.15:
            descriptions.append("適度な音量の変化がある")
        else:
            descriptions.append("音量の変化が大きくダイナミックな演奏")
    
    # 長さ
    duration = features.get("duration", 0)
    if duration > 0:
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        if duration < 30:
            descriptions.append(f"短い音声（{seconds}秒）")
        elif duration < 60:
            descriptions.append(f"短めの音声（{seconds}秒）")
        elif duration < 180:
            descriptions.append(f"標準的な長さ（{minutes}分{seconds}秒）")
        else:
            descriptions.append(f"長い音声（{minutes}分{seconds}秒）")
    
    # ゼロクロッシングレート（音の質感）
    zcr = features.get("zero_crossing_rate", 0)
    if zcr > 0:
        if zcr < 0.05:
            descriptions.append("滑らかで持続的な音")
        elif zcr < 0.15:
            descriptions.append("通常の音質")
        else:
            descriptions.append("ノイズやパーカッシブな要素を含む")

    # 調性を説明に反映
    key = features.get("key") or ""
    if key and mode:
        descriptions.append(f"調性: {key}{'長調' if mode=='major' else '短調'}")
    elif mode:
        descriptions.append(f"{'長調' if mode=='major' else '短調'}の傾向")

    return "。".join(descriptions) + "。" if descriptions else "音声ファイル"

def infer_vocals(features: Dict, transcription: Optional[Dict]) -> Dict:
    """楽曲に歌声があるか、ボーカロイド風かを簡易推定（誤検出を減らすため厳しめ）"""
    vocals = False
    vocaloid = False
    try:
        tempo = float(features.get("tempo") or 0)
        beat = int(features.get("beat_count") or 0)
        brightness = float(features.get("brightness_score") or 0)
        zcr = float(features.get("zero_crossing_rate") or 0)
        f0 = float(features.get("pitch_mean") or 0)
        text = ((transcription or {}).get("text") or "").strip()
        conf = float((transcription or {}).get("confidence") or 0.0)

        # Whisperテキスト根拠（より厳格）
        has_reliable_lyrics = (len(text) >= 20 and conf >= 0.6)
        if has_reliable_lyrics:
            vocals = True

        # ピッチ/テンポ/明度/ゼロクロからの推定（範囲を絞って誤検出を抑制）
        # 一般的な歌声の基本周波数帯に限定し、ビートと明度は高めを要求、ZCRは中庸域に限定
        if (120 <= f0 <= 350) and (beat >= 12) and (brightness >= 0.50) and (0.06 <= zcr <= 0.14):
            vocals = True

        # ボーカロイド風（高めのF0かつ低ZCR、文字起こしは短く信頼低め）
        if f0 >= 220 and zcr <= 0.10 and len(text) < 15 and conf < 0.6:
            vocaloid = True
            vocals = True
    except Exception:
        pass
    return {"vocals": vocals, "vocaloid": vocaloid}

def generate_tags_from_features(features: Dict) -> List[str]:
    """音響特徴量からタグを生成"""
    tags = []
    mode = features.get("mode", "")
    brightness = features.get("brightness_score", 0.0)
    audio_type = features.get("audio_type", "")

    # 会話優先: 音楽的タグを避ける
    if audio_type == "speech":
        duration = features.get("duration", 0)
        if duration > 0:
            if duration < 60:
                tags.extend(["短い", "ショート"])
            elif duration < 180:
                tags.append("標準")
            else:
                tags.extend(["長い", "ロング"])
        tags.extend(["会話", "トーク", "スピーチ", "音声", "audio"])
        return list(set(tags))

    # 効果音（SFX）: シンプルにSFX系のタグのみ
    if audio_type == "sfx":
        duration = features.get("duration", 0)
        if duration > 0:
            if duration < 3:
                tags.extend(["ワンショット", "短い", "ショート"])
            elif duration < 10:
                tags.extend(["短め"])
        tags.extend(["効果音", "sfx", "音声", "audio"])
        return list(set(tags))

    # テンポタグ（補正後を優先）
    tempo = features.get("tempo_refined", features.get("tempo", 0))
    if tempo > 0:
        if tempo < 60:
            tags.extend(["ゆったり", "スロー", "relaxing"])
        elif tempo < 90:
            tags.extend(["ミディアムスロー", "落ち着いた"])
        elif tempo < 120:
            tags.extend(["ミディアムテンポ", "中速"])
        elif tempo < 140:
            tags.extend(["軽快", "アップテンポ"])
        else:
            # マイナーかつ暗めの場合は "energetic" を避ける
            if mode == "minor" and brightness < 0.45:
                tags.extend(["速い", "ハイテンポ"])  
            else:
                tags.extend(["速い", "ハイテンポ", "energetic"])
        tags.append(f"{int(tempo)}BPM")
    
    # 音色タグ
    # 明暗タグ
    if brightness > 0:
        if brightness < 0.35:
            tags.extend(["ダーク", "重厚", "暗い"])
        elif brightness < 0.6:
            tags.extend(["バランス"])
        else:
            if mode == "minor":
                tags.extend(["明るさあり", "陰影"])
            else:
                tags.extend(["明るい", "クリア", "シャープ"])
    
    # ダイナミクスタグ
    rms_std = features.get("rms_std", 0)
    if rms_std > 0:
        if rms_std < 0.05:
            tags.extend(["安定", "一定"])
        elif rms_std < 0.15:
            tags.extend(["適度な変化"])
        else:
            tags.extend(["ダイナミック", "変化に富む"])
    
    # 長さタグ
    duration = features.get("duration", 0)
    if duration > 0:
        if duration < 30:
            tags.extend(["短い", "ショート"])
        elif duration < 60:
            tags.extend(["短め"])
        elif duration < 180:
            tags.extend(["標準"])
        else:
            tags.extend(["長い", "ロング"])
        
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        if minutes > 0:
            tags.append(f"{minutes}分{seconds}秒")
        else:
            tags.append(f"{seconds}秒")
    
    # 音質タグ
    zcr = features.get("zero_crossing_rate", 0)
    if zcr > 0:
        if zcr < 0.05:
            tags.extend(["滑らか", "持続音"])
        elif zcr > 0.15:
            tags.extend(["パーカッシブ", "リズミカル"])
    
    # 調性タグ
    key = features.get("key") or ""
    if key:
        tags.extend([key])
    if mode:
        tags.extend(["長調" if mode == "major" else "短調", mode])

    # ムード極性タグ
    if brightness < 0.4 or (mode == "minor" and brightness < 0.55):
        tags.extend(["暗め", "mood:dark"])
    elif brightness > 0.6 and mode == "major":
        tags.extend(["明るめ", "mood:bright"])
    else:
        tags.append("mood:neutral")

    # 長尺・低ZCR・中庸な明度 → 器楽（オーケストラ/クラシック）傾向
    try:
        _dur = float(features.get("duration") or 0)
        _zcr = float(features.get("zero_crossing_rate") or 0)
    except Exception:
        _dur, _zcr = 0.0, 0.0
    if _dur >= 600 and _zcr <= 0.06 and 0.25 <= float(brightness) <= 0.7:
        tags.extend(["器楽", "オーケストラ", "クラシック"])

    # 基本タグ
    tags.extend(["audio", "音楽"])  # "音声" は会話/SFXで付与済み
    
    return list(set(tags))  # 重複を除去

def infer_instruments(features: Dict) -> List[str]:
    """簡易ヒューリスティクスで楽器群を推定"""
    instruments: List[str] = []
    brightness = float(features.get("brightness_score") or 0.0)
    zcr = float(features.get("zero_crossing_rate") or 0.0)
    tempo = float(features.get("tempo_refined", features.get("tempo", 0.0)) or 0.0)
    onset_rate = float(features.get("onset_rate") or 0.0)
    harm = float(features.get("harmonic_ratio") or 0.0)
    perc = float(features.get("percussive_ratio") or 0.0)
    flat = float(features.get("spectral_flatness_mean") or 0.0)
    duration = float(features.get("duration") or 0.0)
    tonal_presence = float(features.get("tonal_presence") or 0.0)

    # 打楽器/ドラム
    if perc >= 0.35 and onset_rate >= 0.8:
        instruments.append("打楽器")
        instruments.append("ドラム")
    elif perc >= 0.25 and onset_rate >= 0.5:
        instruments.append("打楽器")

    # 弦楽器（ストリングス）: 持続音優勢・低ZCR・明度は広め許容、長尺 or トーナル強い
    if harm >= 0.55 and zcr <= 0.10 and 0.15 <= brightness <= 0.7 and (duration >= 90 or tonal_presence >= 0.22):
        instruments.append("ストリングス")
        instruments.append("弦楽器")

    # 金管: 明るめ・ZCR中庸・打楽器比率低〜中
    if brightness >= 0.65 and 0.06 <= zcr <= 0.14 and perc <= 0.4:
        instruments.append("金管")

    # 木管: 明度中庸・ZCR低・持続音
    if 0.35 <= brightness <= 0.6 and zcr <= 0.08 and harm >= 0.55:
        instruments.append("木管")

    # ピアノ: 打鍵あり（onset中〜高）かつ持続音もあり、明度中庸
    if 0.2 <= perc <= 0.6 and onset_rate >= 0.6 and 0.3 <= brightness <= 0.7:
        instruments.append("ピアノ")

    # オルガン/持続系: 低ZCR・持続音優勢・明度中庸・オンセット少
    if harm >= 0.7 and zcr <= 0.05 and onset_rate <= 0.3 and 0.3 <= brightness <= 0.7:
        instruments.append("オルガン")

    # ストリングセクション内の主旋律推定（ヴァイオリン/ヴィオラ/チェロ）
    f0_med = float(features.get("f0_median") or 0.0)
    f0_iqr = float(features.get("f0_iqr") or 0.0)
    if harm >= 0.5 and zcr <= 0.12 and onset_rate <= 1.2:
        # f0帯域でざっくり分類（中央値/IQRが極端に大きすぎない）
        if 300.0 <= f0_med <= 1000.0 and f0_iqr <= 400.0:
            instruments.append("ヴァイオリン")
        elif 200.0 <= f0_med < 300.0 and f0_iqr <= 300.0:
            instruments.append("ヴィオラ")
        elif 65.0 <= f0_med < 200.0 and f0_iqr <= 250.0:
            instruments.append("チェロ")

    # オーケストラ的構成（広め条件）
    if duration >= 300 and harm >= 0.5 and perc <= 0.55:
        if "ストリングス" not in instruments:
            instruments.append("ストリングス")
        if 0.5 <= brightness <= 0.8 and "金管" not in instruments:
            instruments.append("金管")
        if 0.35 <= brightness <= 0.65 and "木管" not in instruments:
            instruments.append("木管")

    # ユニーク化
    uniq: List[str] = []
    seen = set()
    for inst in instruments:
        if inst not in seen:
            uniq.append(inst)
            seen.add(inst)
    return uniq

def normalize_genre_label(label: str) -> str:
    l = (label or '').strip().lower()
    if not l:
        return ''
    # English/Japanese common forms to categories
    mapping = {
        'pop': 'ポップ', 'pops': 'ポップ', 'ポップ': 'ポップ',
        'orchestra': 'オーケストラ', 'orchestral': 'オーケストラ', 'classical': 'クラシック', 'クラシック': 'クラシック', 'オーケストラ': 'オーケストラ',
        'jazz': 'ジャズ', 'ジャズ': 'ジャズ',
        'rock': 'ロック', 'ロック': 'ロック',
        'electronic': 'エレクトロニック', 'edm': 'エレクトロニック', 'techno': 'エレクトロニック', 'house': 'エレクトロニック', 'エレクトロニック': 'エレクトロニック',
        'ambient': 'アンビエント', 'アンビエント': 'アンビエント',
        'hip hop': 'ヒップホップ', 'hiphop': 'ヒップホップ', 'ヒップホップ': 'ヒップホップ'
    }
    # direct match
    if l in mapping:
        return mapping[l]
    # contains
    for k, v in mapping.items():
        if k in l:
            return v
    return ''

def infer_category(features: Dict, instruments: List[str], voice_present: bool) -> str:
    """ヒューリスティクスで大まかなカテゴリ（ポップ/オーケストラ/ジャズ/ロック/エレクトロニック/アンビエント等）を推定"""
    brightness = float(features.get('brightness_score') or 0.0)
    zcr = float(features.get('zero_crossing_rate') or 0.0)
    tempo = float(features.get('tempo_refined', features.get('tempo', 0.0)) or 0.0)
    onset_rate = float(features.get('onset_rate') or 0.0)
    harm = float(features.get('harmonic_ratio') or 0.0)
    perc = float(features.get('percussive_ratio') or 0.0)
    duration = float(features.get('duration') or 0.0)

    inst = set(instruments or [])
    has_melody = bool(features.get('has_melody'))

    # Orchestra/Classical（メロディ要件を緩和、楽器と持続音優勢を重視）
    if (not voice_present) and duration >= 300 and harm >= 0.5 and perc <= 0.55 and (('ストリングス' in inst) or ('木管' in inst) or ('金管' in inst)):
        return 'オーケストラ'
    if (not voice_present) and duration >= 300 and harm >= 0.6 and zcr <= 0.07 and 'ストリングス' in inst:
        return 'クラシック'

    # Jazz: brass + piano, mid tempo 90-180, moderate onset
    if ('金管' in inst and 'ピアノ' in inst) and 90 <= tempo <= 180 and 0.3 <= onset_rate <= 1.2:
        return 'ジャズ'

    # Rock: high percussive, high onset, bright
    if perc >= 0.5 and onset_rate >= 1.0 and brightness >= 0.5:
        return 'ロック'

    # Electronic: high flatness or synthetic traits, moderate-high brightness, beats present
    flat = float(features.get('spectral_flatness_mean') or 0.0)
    if flat >= 0.35 and tempo >= 90 and brightness >= 0.5:
        return 'エレクトロニック'

    # Pop: vocals often present, tempo 80-140, percussive moderate, brightness mid-high
    if voice_present and 80 <= tempo <= 140 and 0.2 <= perc <= 0.6 and brightness >= 0.4:
        return 'ポップ'

    # Ambient: long, low onset, low percussive, low zcr or very smooth（ただしオーケストラ条件に該当すれば優先）
    if duration >= 300 and onset_rate <= 0.3 and perc <= 0.3 and zcr <= 0.06:
        return 'アンビエント'

    return ''

def transcribe_with_whisper(
    audio_path: Path,
    model_size: str = "base",
    max_seconds: Optional[float] = None,
    offset_seconds: float = 0.0,
    verbose: Optional[bool] = None
) -> Optional[Dict]:
    """Whisperで音声を文字起こし"""
    try:
        safe_debug_print(f"[1/3] Whisperモデル({model_size})を読み込み中...")
        start_time = time.time()
        # キャッシュディレクトリを明示的に指定
        download_root = os.path.expanduser("~/.cache/whisper")
        model = whisper.load_model(model_size, download_root=download_root)
        safe_debug_print(f"Whisperモデル読み込み完了: {time.time() - start_time:.1f}秒")
        
        safe_debug_print(f"[2/3] 文字起こし開始: {audio_path}")
        start_time = time.time()
        # 言語は自動検出（固定しない）
        # 長尺対策: 指定があれば一部のみを読み込み
        audio_input = None
        target_seconds = None
        if (max_seconds and max_seconds > 0) or (offset_seconds and offset_seconds > 0):
            y, sr = _safe_load_audio(str(audio_path), target_sr=16000,
                                     offset=max(0.0, float(offset_seconds or 0.0)),
                                     duration=max_seconds if (max_seconds and max_seconds > 0) else None)
            if y is not None:
                audio_input = y
                try:
                    target_seconds = float(len(y)) / float(sr or 16000)
                except Exception:
                    target_seconds = None
                safe_debug_print(f"Whisper入力を部分読込(ffmpeg対応): offset={offset_seconds}s, duration={max_seconds}s, len={len(y)}")
            else:
                safe_debug_print("部分読込に失敗。ファイル全体で実行")
                audio_input = None
        if target_seconds is None:
            try:
                import librosa as _lib
                target_seconds = float(_lib.get_duration(path=str(audio_path)))
            except Exception:
                target_seconds = None
        # verbose: True でセグメント進捗を標準出力に出す
        if verbose is None:
            import os as _os
            verbose = str(_os.getenv('WHISPER_VERBOSE') or _os.getenv('AUDIO_PROGRESS') or '0').lower() in ('1','true','yes','on')
        # 進捗/ETAロガー
        import os as _os
        show_progress = str(_os.getenv('AUDIO_PROGRESS') or '0').lower() in ('1','true','yes','on')
        interval = float(_os.getenv('WHISPER_PROGRESS_INTERVAL') or 2.0)
        rtf_env = _os.getenv('WHISPER_RTF')
        def _rtf_default(sz: str) -> float:
            s = (sz or '').lower()
            if 'tiny' in s: return 0.6
            if 'base' in s: return 1.0
            if 'small' in s: return 1.6
            if 'medium' in s: return 2.5
            if 'large' in s: return 4.0
            return 2.0
        rtf = float(rtf_env) if rtf_env else _rtf_default(model_size)
        stop_event = None
        start_ts = time.time()
        if show_progress and target_seconds and target_seconds > 0:
            stop_event = threading.Event()
            def _worker():
                while not stop_event.is_set():
                    elapsed = time.time() - start_ts
                    expected = max(1e-6, target_seconds * rtf)
                    pct = min(99.0, max(0.0, (elapsed / expected) * 100.0))
                    eta = max(0.0, expected - elapsed)
                    mm = int(eta // 60)
                    ss = int(eta % 60)
                    safe_debug_print(f"[B] Whisper進捗: {pct:5.1f}% ETA {mm:02d}:{ss:02d}")
                    stop_event.wait(interval)
            threading.Thread(target=_worker, daemon=True).start()
        # 言語/タスクのヒント
        lang_hint = (_os.getenv('WHISPER_LANGUAGE') or '').strip() or None
        task_hint = (_os.getenv('WHISPER_TASK') or 'transcribe').strip() or 'transcribe'
        try:
            result = model.transcribe(
                audio_input if audio_input is not None else str(audio_path),
                fp16=False,
                verbose=verbose,
                language=lang_hint,
                task=task_hint
            )
        finally:
            if stop_event is not None:
                stop_event.set()
                safe_debug_print(f"[B] Whisper進捗: 100.0% ETA 00:00")
        safe_debug_print(f"[3/3] 文字起こし処理完了: {time.time() - start_time:.1f}秒")
        
        # 結果を整形（no_speech_probを保持し、簡易信頼度を算出）
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": (seg.get("text") or "").strip(),
                "no_speech_prob": seg.get("no_speech_prob"),
                "avg_logprob": seg.get("avg_logprob")
            })
        conf = None
        if segments:
            probs = [s.get("no_speech_prob") for s in segments if s.get("no_speech_prob") is not None]
            if probs:
                try:
                    conf = float(1.0 - (sum(probs) / len(probs)))
                except Exception:
                    conf = None
        transcription = {
            "text": (result.get("text") or "").strip(),
            "segments": segments,
            "language": result.get("language"),
            "confidence": conf
        }
        
        safe_debug_print(f"文字起こし完了: {len(transcription['text'])}文字")
        return transcription
        
    except Exception as e:
        safe_debug_print(f"Whisper文字起こしエラー: {e}")
        return None

def estimate_speaker_gender(features: Dict) -> str:
    """ピッチ平均から話者性別の目安を推定（簡易）"""
    try:
        f0 = float(features.get("pitch_mean") or 0.0)
    except Exception:
        f0 = 0.0
    if 85 <= f0 <= 180:
        return "male"
    if 165 <= f0 <= 255:
        return "female"
    return "unknown"

def analyze_audio_comprehensive(
    audio_path: Path,
    use_whisper: bool = True,
    whisper_model: str = "base",
    whisper_max_seconds: Optional[float] = None,
    whisper_offset_seconds: float = 0.0
) -> Dict:
    """音声を総合的に解析"""
    
    safe_debug_print(f"音声ファイルの総合解析開始: {audio_path}")
    
    result = {
        "file_path": str(audio_path),
        "file_name": audio_path.name,
        "features": {},
        "description": "",
        "tags": [],
        "transcription": None
    }
    
    # 1. 音響特徴量を抽出
    safe_debug_print("[A] 音響特徴量を抽出中...")
    # 簡易ETA（音声長×係数）
    feat_stop = None
    try:
        import librosa as _lib
        dur_est = None
        try:
            dur_est = float(_lib.get_duration(path=str(audio_path)))
        except Exception:
            dur_est = None
        import os as _os
        show_progress = str(_os.getenv('AUDIO_PROGRESS') or '0').lower() in ('1','true','yes','on')
        feat_rtf = float(_os.getenv('FEATURES_RTF') or 0.25)  # 0.25s処理/1s音声 目安
        interval = float(_os.getenv('FEATURES_PROGRESS_INTERVAL') or 2.0)
        if show_progress and dur_est and dur_est > 0:
            stop_event = threading.Event()
            start_ts = time.time()
            def _worker():
                while not stop_event.is_set():
                    elapsed = time.time() - start_ts
                    expected = max(1e-6, dur_est * feat_rtf)
                    pct = min(95.0, max(0.0, (elapsed / expected) * 100.0))
                    eta = max(0.0, expected - elapsed)
                    mm = int(eta // 60)
                    ss = int(eta % 60)
                    safe_debug_print(f"[A] 特徴量進捗: {pct:5.1f}% ETA {mm:02d}:{ss:02d}")
                    stop_event.wait(interval)
            threading.Thread(target=_worker, daemon=True).start()
            def _stop():
                stop_event.set()
                safe_debug_print(f"[A] 特徴量進捗: 100.0% ETA 00:00")
            feat_stop = _stop
    except Exception:
        feat_stop = None
    features = extract_audio_features(audio_path)
    if feat_stop:
        try:
            feat_stop()
        except Exception:
            pass
    result["features"] = features
    
    # 2. まずWhisperで音声（声）の有無を推定（オーケストラ誤判定対策として先に実施）
    transcription = None
    if use_whisper:
        safe_debug_print("[B] Whisperで文字起こし中...")
        transcription = transcribe_with_whisper(
            audio_path,
            whisper_model,
            max_seconds=whisper_max_seconds,
            offset_seconds=whisper_offset_seconds,
            verbose=None,
        )
        if transcription:
            result["transcription"] = transcription
            safe_debug_print(f"文字起こし完了（{len(transcription.get('text',''))}文字）")

    # 3. 声の有無を推定（Whisper結果とセグメントのno_speech_probから）
    def _estimate_voice_presence(feat: Dict, tr: Optional[Dict]) -> bool:
        if not tr:
            return False
        text = (tr.get("text") or "").strip()
        conf = float(tr.get("confidence") or 0.0)
        if conf >= 0.55 and len(text) >= 10:
            return True
        # セグメントから有声区間の総時間を推定
        total_dur = float(feat.get("duration") or 0.0)
        voiced = 0.0
        segs = tr.get("segments") or []
        for s in segs:
            try:
                ns = s.get("no_speech_prob")
                if ns is None or float(ns) <= 0.5:
                    start = float(s.get("start") or 0.0)
                    end = float(s.get("end") or 0.0)
                    if end > start:
                        voiced += (end - start)
            except Exception:
                continue
        if total_dur > 0 and voiced / total_dur >= 0.2 and voiced >= 4.0:
            return True
        return False

    voice_present = _estimate_voice_presence(features, result.get("transcription"))
    features["voice_present"] = bool(voice_present)

    # 3.2 種別判定（会話/音楽）— 声の有無を考慮して優先的に会話を識別
    audio_type = classify_audio_type(features, result.get("transcription")) if features else "ambiguous"
    # フォールバック：声がはっきりあるのに曖昧/音楽扱いなら会話に寄せる
    if voice_present and audio_type in ("ambiguous", "music"):
        # 強い音楽証拠（テンポ・ビート・メロディ）のときのみ音楽を維持
        tempo = float(features.get("tempo_refined", features.get("tempo") or 0))
        beat = int(features.get("beat_count") or 0)
        mel = bool(features.get("has_melody"))
        # バラード等への配慮: メロディがあり、十分な歌詞があるなら低テンポでも音楽扱い
        tr_text = ((result.get("transcription") or {}).get("text") or "").strip()
        strong_music = (mel and (tempo >= 100 or beat >= 10)) or (mel and len(tr_text) >= 20)
        if not strong_music:
            audio_type = "speech"
    features["audio_type"] = audio_type

    # CREPEでメロディ補強（任意）
    try:
        y_repr, sr_repr = _load_repr_segment(audio_path, target_sr=16000)
        crepe_res = _detect_melody_with_crepe(y_repr, sr_repr)
        if crepe_res:
            features.update({
                'crepe_voiced_ratio': crepe_res.get('crepe_voiced_ratio'),
                'crepe_longest_run_sec': crepe_res.get('crepe_longest_run_sec'),
                'crepe_f0_median': crepe_res.get('crepe_f0_median')
            })
            if crepe_res.get('crepe_has_melody'):
                features['has_melody'] = True
    except Exception:
        pass

    # 4. 説明文とタグを生成（種別に応じて）
    if features:
        if audio_type == "speech":
            # 会話向けの説明
            lang = (result.get("transcription") or {}).get("language") or "ja"
            desc = "会話主体の音声"
            if lang:
                desc += f"（言語: {lang}）"
            # 長さ
            duration = features.get("duration")
            if duration:
                m = int(duration // 60)
                s = int(duration % 60)
                if m > 0:
                    desc += f"。長さ: {m}分{s}秒"
                else:
                    desc += f"。長さ: {s}秒"
            # 抜粋（歌詞テキストが存在するなら必ず一部を追加）
            tr = result.get("transcription") or {}
            text_full = (tr.get("text") or "").strip()
            if len(text_full) >= 1:
                snippet = text_full[:200].strip()
                if snippet:
                    desc += f"。内容抜粋: 『{snippet}』"
            result["description"] = desc
            result["tags"] = generate_tags_from_features(features)
            # 話者性別（簡易）
            gender = estimate_speaker_gender(features)
            result["speaker_gender"] = gender
            if gender == "male":
                result["tags"].extend(["男性声"])
            elif gender == "female":
                result["tags"].extend(["女性声"])

            # 付加: 背景に音楽性がある場合はBGM要約を併記（曲の内容も提示）
            try:
                has_mel = bool(features.get("has_melody"))
                tonal = float(features.get("tonal_presence") or 0.0)
                tempo_ref = float(features.get("tempo_refined", features.get("tempo", 0.0)) or 0.0)
                if has_mel or tonal >= 0.18 or tempo_ref > 0:
                    bgm_desc = features_to_description(features)
                    if bgm_desc:
                        short_bgm = bgm_desc if len(bgm_desc) <= 140 else (bgm_desc[:140] + "…")
                        # 楽器/カテゴリの簡易推定を添える
                        insts = infer_instruments(features)
                        cat = infer_category(features, insts, voice_present)
                        if cat:
                            short_bgm += f"（カテゴリ: {cat}）"
                        result["description"] += f"。BGM推定: {short_bgm}"
                        # タグにBGM/楽器/カテゴリを少量追加
                        if insts:
                            result["tags"].extend(insts[:3])
                        if cat:
                            result["tags"].append(cat)
                        result["tags"].append("BGM")
            except Exception:
                pass
        elif audio_type == "sfx":
            # 効果音向けの説明
            desc = "効果音"
            duration = features.get("duration")
            if duration:
                m = int(duration // 60)
                s = int(duration % 60)
                if m > 0:
                    desc += f"（{m}分{s}秒）"
                else:
                    desc += f"（{s}秒）"
            zcr = features.get("zero_crossing_rate")
            if zcr is not None:
                if zcr > 0.15:
                    desc += "。パーカッシブ/ノイズ感"
                elif zcr < 0.05:
                    desc += "。滑らか/持続音"
            result["description"] = desc
            result["tags"] = generate_tags_from_features(features)
        else:
            # 楽曲解析（Whisperとは分離）
            music_desc = features_to_description(features)
            result["music_description"] = music_desc
            # まずは楽曲解析をキャプションの先頭に据える
            base_desc = f"楽曲解析: {music_desc}" if music_desc else "楽曲解析"
            result["description"] = base_desc
            result["tags"] = generate_tags_from_features(features)
            # 楽器推定（ヒューリスティクス）
            insts = infer_instruments(features)
            # PANNsでの楽器補強（任意・ネットワーク未使用時はスキップ）
            import os as _os
            # 既定でPANNsを使用。明示的に DISABLE_PANNS=1 または NO_HF_DOWNLOAD=1 の場合のみスキップ
            _disable_panns = str(_os.getenv('DISABLE_PANNS') or '').lower() in ('1','true')
            _skip_panns = _disable_panns or str(_os.getenv('NO_HF_DOWNLOAD') or '').lower() in ('1','true')
            if not _skip_panns:
                pann_insts = _infer_instruments_with_panns(audio_path)
                if pann_insts:
                    insts = list(set(insts + pann_insts))
            if insts:
                result.setdefault("instruments", insts)
                result["tags"].extend(insts)
            # カテゴリ推定（フォールバック）
            category = infer_category(features, insts, voice_present)
            if category:
                result["category"] = category
                result["tags"].append(category)
            # 楽曲の歌声推定（声の有無を最優先。タグ付けは更に厳格: 信頼できる歌詞 or 明確なボカロ判定時のみ）
            vocal_info = infer_vocals(features, result.get("transcription"))
            tr = result.get("transcription") or {}
            _conf = float(tr.get("confidence") or 0.0)
            _text = (tr.get("text") or "").strip()
            if voice_present and vocal_info.get("vocals") and ( (_conf >= 0.6 and len(_text) >= 12) or vocal_info.get("vocaloid") ):
                result["description"] += " 歌声あり"
                result["tags"].extend(["ボーカル", "歌声"])
            if voice_present and vocal_info.get("vocaloid"):
                result["description"] += "（ボーカロイド風）"
                result["tags"].extend(["ボーカロイド", "合成音声"]) 
            # 歌詞は分離して併記（LLMに渡さない）
            tr = result.get("transcription") or {}
            conf = tr.get("confidence") or 0.0
            text_full = (tr.get("text") or "").strip()
            if len(text_full) >= 1:
                snippet = text_full[:200]
                result["lyrics_snippet"] = snippet
                result["description"] += f"。歌詞抜粋: 『{snippet}』"
            # タグ付けは従来通りの閾値で（誤タグを避けるため）
            lyric_threshold = 0.4 if vocal_info.get("vocals") else 0.6
            if conf >= lyric_threshold and len(text_full) >= 12:
                result["tags"].extend(["歌詞あり", "ボーカル", "vocal"])
            # 歌声の性別は推定しづらいため省略（合成音声は別タグで表示）
            # 楽曲などでも声があれば性別推定
            gender = estimate_speaker_gender(features)
            if gender in ("male", "female"):
                result["speaker_gender"] = gender
                result["tags"].extend(["男性声"] if gender == "male" else ["女性声"])

            # キャプション補足: メロディ/歌詞/カテゴリ
            melody_flag = "あり" if features.get("has_melody") else "なし"
            # 歌詞フラグは『(一定長の文字起こし) or (有声/歌声推定)』で一貫させる
            text_full = (result.get("transcription", {}) or {}).get("text", "")
            has_text_lyrics = bool(text_full and len(text_full.strip()) >= 12)
            # Whisperの信頼度が低くても音響的に歌声が強い場合は「歌詞:あり」とみなす
            lyrics_flag = "あり" if (has_text_lyrics or voice_present or vocal_info.get("vocals")) else "なし"
            if category:
                result["description"] += f"。カテゴリ: {category}"
            result["description"] += f"。メロディ: {melody_flag}／歌詞: {lyrics_flag}"

        # --- 完全分離: 最後に必ず「楽曲解析」→「歌詞抜粋」の順でキャプションを構成 ---
        try:
            # 楽器/カテゴリを再計算（どの分岐でも一定の基準で表示）
            insts_final = infer_instruments(features)
            import os as _os
            _disable_panns = str(_os.getenv('DISABLE_PANNS') or '').lower() in ('1','true')
            _skip_panns = _disable_panns or str(_os.getenv('NO_HF_DOWNLOAD') or '').lower() in ('1','true')
            if not _skip_panns:
                pann_insts = _infer_instruments_with_panns(audio_path)
                if pann_insts:
                    for i in pann_insts:
                        if i not in insts_final:
                            insts_final.append(i)
            category_final = infer_category(features, insts_final, bool(features.get('voice_present')))

            music_desc_final = features_to_description(features)
            result['music_description'] = music_desc_final
            music_sec = f"楽曲解析: {music_desc_final}" if music_desc_final else "楽曲解析: 音楽的要素は弱い/未検出"
            if category_final:
                music_sec += f"（カテゴリ: {category_final}）"

            # 歌詞抜粋（Whisperテキスト）
            tr = result.get('transcription') or {}
            snippet = ((tr.get('text') or '').strip())[:200]
            lyrics_sec = f"歌詞抜粋: 『{snippet}』" if snippet else ""

            # 最終キャプション
            result['description'] = music_sec + (f"。{lyrics_sec}" if lyrics_sec else "")

            # タグ統合（重複回避）
            if insts_final:
                for t in insts_final:
                    if t not in result['tags']:
                        result['tags'].append(t)
            if category_final and category_final not in result['tags']:
                result['tags'].append(category_final)
        except Exception:
            pass

        safe_debug_print(f"生成されたタグ: {', '.join(result['tags'][:10])}")
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python audio_analyzer_v2.py <音声ファイルパス>")
        sys.exit(1)
    
    audio_file = Path(sys.argv[1])
    if not audio_file.exists():
        print(f"ファイルが見つかりません: {audio_file}")
        sys.exit(1)
    
    # 解析実行
    result = analyze_audio_comprehensive(audio_file, use_whisper=True)
    
    # 結果を表示
    print("\n=== 解析結果 ===")
    print(f"ファイル: {result['file_name']}")
    print(f"\n説明: {result['description']}")
    print(f"\nタグ: {', '.join(result['tags'])}")
    
    if result.get('transcription'):
        print(f"\n文字起こし: {result['transcription']['text'][:200]}...")
    
    # JSON形式でも出力
    print(f"\n=== JSON出力 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
