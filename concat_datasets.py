import pandas as pd
from datasets import load_dataset

# 1. Локальный файл 1
df1 = pd.read_csv('/Users/maksimkuznetsov/Desktop/xTRam1_combined_file.csv')
df1 = df1[['text', 'label']].copy()

# 2. HuggingFace Dataset 1
ds2 = load_dataset('hendzh/PromptShield', split='train')
df2 = pd.DataFrame(ds2)
df2 = df2.rename(columns={'prompt': 'text'})[['text', 'label']]

# 3. HuggingFace Dataset 2
df3 = pd.read_csv("/Users/maksimkuznetsov/Downloads/prompt-injections-benchmark.csv")
df3['label'] = df3['label'].map({'benign': 0, 'jailbreak': 1})
df3 = df3[['text', 'label']]

# 4. HuggingFace Dataset 3
ds4 = load_dataset('jackhhao/jailbreak-classification', split='train')
df4 = pd.DataFrame(ds4)
df4['label'] = df4['type'].map({'benign': 0, 'jailbreak': 1})
df4 = df4.rename(columns={'prompt': 'text'})[['text', 'label']]

# 5. Локальный файл 2
df5 = pd.read_csv('/Users/maksimkuznetsov/Downloads/synthetic_dataset.csv')
df5 = df5.rename(columns={
    'prompt': 'text',
    'jailbreak': 'label'
})[['text', 'label']]

# Объединение всех датасетов
combined_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# Удаление дубликатов и пустых значений
combined_df = combined_df.dropna().drop_duplicates()

# Сохранение результата
combined_df.to_csv('combined_jailbreak_dataset.csv', index=False)

print(f"Объединенный датасет сохранен. Размер: {combined_df.shape}")