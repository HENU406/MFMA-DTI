
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings('ignore')

class PretrainedFeatureGenerator:
    """使用预训练模型生成药物和蛋白质特征"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        self.drug_tokenizer = None
        self.drug_model = None
        self.protein_tokenizer = None
        self.protein_model = None

    def load_drug_model(self, model_name="./models/chemberta"):
        if self.drug_model is not None:
            return

        print("\n" + "=" * 70)
        print("加载 ChemBERTa 模型...")
        print("=" * 70)

        try:
            print(f"模型: {model_name}")

            self.drug_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.drug_model = AutoModel.from_pretrained(model_name)
            self.drug_model.to(self.device)
            self.drug_model.eval()

            # 测试
            test_input = self.drug_tokenizer("CC(=O)O", return_tensors="pt")
            test_input = {k: v.to(self.device) for k, v in test_input.items()}
            with torch.no_grad():
                test_output = self.drug_model(**test_input)

            feature_dim = test_output.last_hidden_state.shape[-1]
            print(f"ChemBERTa 加载完成!")
            print(f"   特征维度: {feature_dim}")

        except Exception as e:
            print(f"加载失败: {e}")
            print("\n尝试备用模型...")

            # 备用模型列表
            backup_models = [
                "DeepChem/ChemBERTa-77M-MLM",
                "seyonec/PubChem10M_SMILES_BPE_450k",
            ]

            for backup in backup_models:
                try:
                    print(f"尝试: {backup}")
                    self.drug_tokenizer = AutoTokenizer.from_pretrained(backup)
                    self.drug_model = AutoModel.from_pretrained(backup)
                    self.drug_model.to(self.device)
                    self.drug_model.eval()
                    print(f"备用模型加载成功!")
                    break
                except:
                    continue
            else:
                raise Exception("所有ChemBERTa模型都加载失败")

    def load_protein_model(self, model_name="./models/protbert"):
        """加载ProtBERT模型"""
        if self.protein_model is not None:
            return

        print("\n" + "=" * 70)
        print("加载 ProtBERT 模型...")
        print("=" * 70)

        try:
            print(f"模型: {model_name}")

            self.protein_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                do_lower_case=False
            )
            self.protein_model = AutoModel.from_pretrained(model_name)
            self.protein_model.to(self.device)
            self.protein_model.eval()

            # 测试
            test_seq = " ".join("MKTII")
            test_input = self.protein_tokenizer(test_seq, return_tensors="pt")
            test_input = {k: v.to(self.device) for k, v in test_input.items()}
            with torch.no_grad():
                test_output = self.protein_model(**test_input)

            feature_dim = test_output.last_hidden_state.shape[-1]
            print(f"ProtBERT 加载完成!")
            print(f"   特征维度: {feature_dim}")

        except Exception as e:
            print(f"加载失败: {e}")
            raise

    def read_smiles(self, smiles_file):
        """读取SMILES文件 (修改版: 增强分隔符兼容性)"""
        drug_ids = []
        smiles_list = []

        print(f"\n读取SMILES文件: {smiles_file}")

        with open(smiles_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line: continue

                # 修改1: 自动处理分隔符 (split() 不带参数会自动处理空格和Tab)
                # maxsplit=1 确保SMILES内部如果有空格（极少见但为了安全）不会被切断
                parts = line.split(maxsplit=1)

                if len(parts) >= 2:
                    drug_id, smiles = parts[0], parts[1]
                    # 跳过无效的SMILES
                    if smiles and smiles not in ['N/A', 'NA', '']:
                        drug_ids.append(drug_id)
                        smiles_list.append(smiles)
                    else:
                        print(f"第 {line_idx + 1} 行跳过: SMILES无效 - {line}")
                else:
                    # 修改2: 增加日志，告诉你哪一行格式不对被跳过了
                    print(f"第 {line_idx + 1} 行跳过: 格式无法解析 - {line}")

        print(f"读取了 {len(drug_ids)} 个有效的药物SMILES")
        return drug_ids, smiles_list

    def read_sequences(self, fasta_file):
        """读取FASTA格式的蛋白质序列"""
        protein_ids = []
        sequences = []

        print(f"\n读取FASTA文件: {fasta_file}")

        current_id = None
        current_seq = []

        with open(fasta_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # 保存上一个序列
                    if current_id is not None:
                        seq = ''.join(current_seq)
                        if seq:  # 确保序列非空
                            protein_ids.append(current_id)
                            sequences.append(seq)

                    # 开始新序列
                    current_id = line[1:]
                    current_seq = []
                else:
                    current_seq.append(line)

            # 保存最后一个序列
            if current_id is not None:
                seq = ''.join(current_seq)
                if seq:
                    protein_ids.append(current_id)
                    sequences.append(seq)

        print(f"读取了 {len(protein_ids)} 个蛋白质序列")

        # 显示序列长度统计
        lengths = [len(seq) for seq in sequences]
        print(f"序列长度: 最小={min(lengths)}, 最大={max(lengths)}, 平均={sum(lengths) / len(lengths):.0f}")

        return protein_ids, sequences

    def encode_drugs(self, smiles_list, batch_size=32):
        """
        使用ChemBERTa编码药物

        Args:
            smiles_list: SMILES字符串列表
            batch_size: 批次大小

        Returns:
            drug_features: (N, feature_dim) numpy array
        """
        self.load_drug_model()

        print(f"\n编码 {len(smiles_list)} 个药物...")
        print(f"批次大小: {batch_size}")

        all_features = []
        failed_indices = []

        with torch.no_grad():
            for i in tqdm(range(0, len(smiles_list), batch_size), desc="编码药物"):
                batch_smiles = smiles_list[i:i + batch_size]

                try:
                    # Tokenize
                    inputs = self.drug_tokenizer(
                        batch_smiles,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Forward
                    outputs = self.drug_model(**inputs)

                    # 使用 [CLS] token 的表示
                    features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    all_features.append(features)

                except Exception as e:
                    print(f"\n批次 {i // batch_size} 编码失败: {e}")
                    # 为失败的批次填充零向量
                    feature_dim = self.drug_model.config.hidden_size
                    failed_features = np.zeros((len(batch_smiles), feature_dim))
                    all_features.append(failed_features)
                    failed_indices.extend(range(i, i + len(batch_smiles)))

        drug_features = np.vstack(all_features)

        print(f"\n药物特征生成完成!")
        print(f"   特征维度: {drug_features.shape}")
        if failed_indices:
            print(f"失败: {len(failed_indices)} 个")

        return drug_features

    def encode_proteins(self, sequences, batch_size=4, max_length=1024):
        """
        使用ProtBERT编码蛋白质

        Args:
            sequences: 蛋白质序列列表
            batch_size: 批次大小（蛋白质序列长，用小batch）
            max_length: 最大序列长度

        Returns:
            protein_features: (N, feature_dim) numpy array
        """
        self.load_protein_model()

        print(f"\n 编码 {len(sequences)} 个蛋白质...")
        print(f"   批次大小: {batch_size}")
        print(f"   最大长度: {max_length}")

        # ProtBERT需要在氨基酸间加空格
        sequences_spaced = [' '.join(list(seq[:max_length])) for seq in sequences]

        all_features = []
        failed_indices = []

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences_spaced), batch_size), desc="编码蛋白质"):
                batch_seqs = sequences_spaced[i:i + batch_size]

                try:
                    # Tokenize
                    inputs = self.protein_tokenizer(
                        batch_seqs,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors='pt'
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Forward
                    outputs = self.protein_model(**inputs)

                    # 使用 [CLS] token 的表示
                    features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    all_features.append(features)

                except Exception as e:
                    print(f"\n批次 {i // batch_size} 编码失败: {e}")
                    # 为失败的批次填充零向量
                    feature_dim = self.protein_model.config.hidden_size
                    failed_features = np.zeros((len(batch_seqs), feature_dim))
                    all_features.append(failed_features)
                    failed_indices.extend(range(i, i + len(batch_seqs)))

        protein_features = np.vstack(all_features)

        print(f"\n蛋白质特征生成完成!")
        print(f"   特征维度: {protein_features.shape}")
        if failed_indices:
            print(f"失败: {len(failed_indices)} 个")

        return protein_features

    def save_features(self, features, ids, output_file):
        """保存特征到文件"""
        np.savez_compressed(
            output_file,
            features=features,
            ids=np.array(ids)
        )
        print(f"特征已保存到: {output_file}")
        print(f"文件大小: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")

    def generate_all_features(self,
                              smiles_file,
                              fasta_file,
                              output_dir,
                              drug_batch_size=32,
                              protein_batch_size=4):
        """
        生成所有特征的主函数

        Args:
            smiles_file: 药物SMILES文件
            fasta_file: 蛋白质FASTA文件
            output_dir: 输出目录
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("预训练特征生成管道")
        print("=" * 70)
        print(f"输出目录: {output_dir}")

        # ==================== 步骤1: 读取数据 ====================
        print("\n" + "=" * 70)
        print("步骤1: 读取原始数据")
        print("=" * 70)

        drug_ids, smiles_list = self.read_smiles(smiles_file)
        protein_ids, sequences = self.read_sequences(fasta_file)

        # ==================== 步骤2: 生成药物特征 ====================
        print("\n" + "=" * 70)
        print("步骤2: 生成药物特征 (ChemBERTa)")
        print("=" * 70)

        drug_features = self.encode_drugs(smiles_list, batch_size=drug_batch_size)

        self.save_features(
            drug_features,
            drug_ids,
            output_dir / 'drug_features.npz'
        )

        # ==================== 步骤3: 生成蛋白质特征 ====================
        print("\n" + "=" * 70)
        print("步骤3: 生成蛋白质特征 (ProtBERT)")
        print("=" * 70)

        protein_features = self.encode_proteins(sequences, batch_size=protein_batch_size)

        self.save_features(
            protein_features,
            protein_ids,
            output_dir / 'protein_features.npz'
        )

        # ==================== 步骤4: 保存映射 ====================
        print("\n" + "=" * 70)
        print("步骤4: 保存ID映射")
        print("=" * 70)

        # 药物ID映射
        with open(output_dir / 'drug_id_mapping.txt', 'w') as f:
            for idx, drug_id in enumerate(drug_ids):
                f.write(f"{idx}\t{drug_id}\n")

        # 蛋白质ID映射
        with open(output_dir / 'protein_id_mapping.txt', 'w') as f:
            for idx, protein_id in enumerate(protein_ids):
                f.write(f"{idx}\t{protein_id}\n")

        print(f" ID映射已保存")

        # ==================== 完成 ====================
        print("\n" + "=" * 70)
        print("特征生成完成!")
        print("=" * 70)
        print(f"\n📁 输出文件:")
        print(f"   - drug_features.npz: {drug_features.shape}")
        print(f"   - protein_features.npz: {protein_features.shape}")
        print(f"   - drug_id_mapping.txt")
        print(f"   - protein_id_mapping.txt")

        print(f"\n 下一步:")
        print(f"   运行数据预处理:")
        print(f"   python preprocess_dti_data.py --data_dir {output_dir}")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='生成预训练特征')
    parser.add_argument('--smiles_file', required=True,
                        help='药物SMILES文件 (drug_id TAB smiles)')
    parser.add_argument('--fasta_file', required=True,
                        help='蛋白质FASTA文件')
    parser.add_argument('--output_dir', required=True,
                        help='输出目录')
    parser.add_argument('--drug_batch_size', type=int, default=32,
                        help='药物编码批次大小')
    parser.add_argument('--protein_batch_size', type=int, default=4,
                        help='蛋白质编码批次大小')
    parser.add_argument('--device', default='auto',
                        help='设备 (cuda/cpu/auto)')

    args = parser.parse_args()

    # 自动选择设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # 创建生成器
    generator = PretrainedFeatureGenerator(device=device)

    # 生成所有特征
    generator.generate_all_features(
        smiles_file=args.smiles_file,
        fasta_file=args.fasta_file,
        output_dir=args.output_dir,
        drug_batch_size=args.drug_batch_size,
        protein_batch_size=args.protein_batch_size
    )


if __name__ == '__main__':
    main()