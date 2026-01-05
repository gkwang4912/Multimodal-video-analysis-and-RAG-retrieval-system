import whisper
import csv
import os

def transcribe_video(video_path, output_csv):
    print(f"正在載入 Whisper 模型...")
    # 使用 base 模型，如果需要更高精度可以改成 'small', 'medium', 'large'
    model = whisper.load_model("large")

    print(f"正在分析影片: {video_path}")
    # 雖然是影片，Whisper 會自動處理音訊部分
    # fp16=False 避免在沒有 GPU 的機器上報錯
    result = model.transcribe(video_path, fp16=False)

    print("分析完成，正在寫入 CSV...")
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['開始時間', '結束時間', '內容']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for segment in result['segments']:
            writer.writerow({
                '開始時間': format_timestamp(segment['start']),
                '結束時間': format_timestamp(segment['end']),
                '內容': segment['text'].strip()
            })
    
    print(f"已成功輸出至: {output_csv}")

def format_timestamp(seconds):
    # 簡單的時間格式化函數 HH:MM:SS
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

if __name__ == "__main__":
    video_file = "test.mp4"
    output_file = "transcript.csv"
    
    if not os.path.exists(video_file):
        print(f"找不到檔案: {video_file}")
    else:
        transcribe_video(video_file, output_file)
