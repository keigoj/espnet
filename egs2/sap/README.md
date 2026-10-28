# SAP レシピ

- **Kaldi データ作成・MFA 文分割:**
  [SAP データ準備手順](asr2/local/README_sap_sentence_split_kaldi.md)
  （必要入力、MFA 環境、実行コマンド、`SAP0430_processed/scripts/` との対応、検証方法）。
- **微分可能な離散トークンを使う ASR:** [asr3/README.md](asr3/README.md)。

データ準備の入口は `asr2/local/run_sap_sentence_split_kaldi.sh`。
`asr3/local` は `asr2/local` へのシンボリックリンクで、同じスクリプトを使用する。
