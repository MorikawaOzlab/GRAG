# GridWorld / BabyAI

既存環境は `GridWorld/` に移動しました。GRAGを含む `agents/` は共有のまま保持し、内容は変更していません。

## BabyAI

```powershell
.\venv\Scripts\python.exe -m pip install -r babyai/requirements.txt
ollama pull gpt-oss:20b
.\babyai_app.bat
```

ブラウザで http://127.0.0.1:5010 を開いてください。Ollamaは起動しておきます。
環境ID、AI、自然言語タスク、最大ステップ、シードを入力して「新しい実験」を押します。
命令が空欄の場合はBabyAI標準の命令・報酬・達成判定を使います。
詳しくは [BabyAI README](babyai/README.md) を参照してください。

## 既存GridWorld

`GridWorld/scripts/training_app.bat`、`map_app.bat`、`settings_app.bat` で起動します。
既存データは `GridWorld/data/` に移動しています。
仮想環境はルートの `venv/` を継続使用します。
