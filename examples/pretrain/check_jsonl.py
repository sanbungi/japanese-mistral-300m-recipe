import orjson
import os
from tqdm import tqdm

def find_broken_jsonl_lines(file_path):
    error_lines = []
    
    # ファイルサイズを取得（tqdmの進捗バー用）
    total_size = os.path.getsize(file_path)
    
    print(f"Processing: {file_path}")
    print(f"Size: {total_size / (1024**3):.2f} GB")

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # tqdmの設定: total=バイト数, unit='B'
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                
                # enumerateで1行ずつ処理 (1始まりで行番号を取得)
                for line_number, line in enumerate(f, 1):
                    # 行のバイト数を進捗バーに反映
                    # (厳密にはエンコードが必要ですが、処理速度優先のため行の長さで近似更新します)
                    pbar.update(len(line.encode('utf-8')))

                    line = line.strip()
                    if not line:
                        continue # 空行はスキップ（必要に応じてエラー扱いにしてください）

                    try:
                        orjson.loads(line)
                    except orjson.JSONDecodeError:
                        # エラー発生時の行番号をリストに追加
                        error_lines.append(line_number)
                        
    except FileNotFoundError:
        print("ファイルが見つかりません。パスを確認してください。")
        return []

    return error_lines

# --- 実行部分 ---
if __name__ == "__main__":
    # ここに対象のファイルパスを指定してください
    FILE_PATH = "./results/dataset/wiki.jsonl" 

    broken_lines = find_broken_jsonl_lines(FILE_PATH)

    print("\n" + "="*30)
    if broken_lines:
        print(f"発見された壊れた行数: {len(broken_lines)}")
        print(f"壊れた行番号 (最初の10件): {broken_lines[:10]}")
        # 全て保存したい場合はファイルに出力推奨
        with open("error_lines.txt", "w") as f:
            for l in broken_lines:
                f.write(f"{l}\n")
        print("全てのエラー行番号を 'error_lines.txt' に保存しました。")
    else:
        print("壊れた行は見つかりませんでした。")