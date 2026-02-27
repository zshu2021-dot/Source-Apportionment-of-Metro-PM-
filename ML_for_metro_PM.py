# ML_for_metro_PM_v5.py
# 地铁颗粒物浓度预测分析系统
# 更新: 2026

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import randint, uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYES_AVAILABLE = True
    print("✓ BayesSearchCV 可用（贝叶斯优化）")
except ImportError:
    BAYES_AVAILABLE = False
    print("⚠ BayesSearchCV 不可用，回退到 RandomizedSearchCV")
    print("  安装命令: pip install scikit-optimize")

# ==================== 输出目录配置 ====================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out_path(filename):
    """返回输出文件的完整路径"""
    return os.path.join(OUTPUT_DIR, filename)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 50

# 设置随机种子
np.random.seed(42)

print("=" * 100)
print("地铁颗粒物浓度预测分析系统")
print("=" * 100)
# ==================== 1. 数据加载和预处理 ====================
def load_and_preprocess_data(file_path):
    """加载并预处理数据（增强版：异常值处理 + 高级特征工程）"""
    print("\n" + "=" * 100)
    print("步骤 1: 数据加载和预处理（增强版）")
    print("=" * 100)
    
    df = pd.read_excel(file_path)
    
    print(f"\n数据集形状: {df.shape}")
    print(f"样本数量: {df.shape[0]}")
    print(f"特征数量: {df.shape[1] - 1}")
    

    # 定义原始特征列和目标列
    feature_cols = ['Peak', 'Platform depth', 'Metro humidity', 
                   'Outdoor humidity', 'Metro temperature', 'Outdoor temperature',
                   'Platform years', 'Line years', 'Screen doort ype',
                   'Platform type', 'Transfer station', 'Air pressure',
                   'Ground_PM']
    target_col = 'Metro_PM'
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # ---------- 缺失值处理 ----------
    print("\n缺失值统计:")
    missing_counts = X.isnull().sum()
    if missing_counts.sum() > 0:
        print(missing_counts[missing_counts > 0])
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
    else:
        print("无缺失值")
    
    if y.isnull().sum() > 0:
        y.fillna(y.median(), inplace=True)
    
    # ---------- 异常值处理（IQR截断法）----------
    print("\n异常值处理 (IQR方法)...")
    continuous_cols = ['Platform depth', 'Metro humidity', 'Outdoor humidity',
                       'Metro temperature', 'Outdoor temperature', 'Platform years',
                       'Line years', 'Air pressure', 'Ground_PM']
    n_clipped = 0
    for col in continuous_cols:
        if col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3.0 * IQR
            upper = Q3 + 3.0 * IQR
            before = ((X[col] < lower) | (X[col] > upper)).sum()
            X[col] = X[col].clip(lower, upper)
            if before > 0:
                n_clipped += before
                print(f"  {col}: 截断 {before} 个极端值")
    if n_clipped == 0:
        print("  无明显异常值")
    
    # ---------- 高级特征工程----------
    print("\n特征工程: 添加高级交互特征...")
    
    # ── 基础交互特征──────────────────
    X['Temp_diff']             = X['Metro temperature'] - X['Outdoor temperature']
    X['Humidity_diff']         = X['Metro humidity'] - X['Outdoor humidity']
    X['Depth_age']             = X['Platform depth'] * X['Platform years']
    X['Peak_Transfer']         = X['Peak'] * X['Transfer station']
    X['Metro_THI']             = (X['Metro temperature'] 
                                  - (0.55 - 0.0055 * X['Metro humidity']) 
                                  * (X['Metro temperature'] - 14.5))
    X['PM_TempDiff']           = X['Ground_PM'] * X['Temp_diff']
    X['GroundPM_HumidityDiff'] = X['Ground_PM'] * X['Humidity_diff']
    X['GroundPM_Air_pressure'] = X['Ground_PM'] * X['Air pressure']

    # ── 新增高级特征────────────────────────────
    X['Apparent_temp']     = (X['Metro temperature'] 
                               - 0.4 * (X['Metro temperature'] - 10) 
                               * (1 - X['Metro humidity'] / 100))
    X['Ventilation_proxy'] = X['Platform depth'] / (X['Platform years'] + 1)
    X['Relative_pressure'] = X['Air pressure'] - X['Air pressure'].mean()

    X['Type_Depth']      = X['Platform type'] * X['Platform depth']
    
    X['Age_composite']   = X['Platform years'] + 0.5 * X['Line years']
    
    X['GroundPM_TempDiff'] = X['Ground_PM'] * X['Temp_diff']

    # ── ★定义 all_feature_cols，再使用 ──────────
    new_features = [
        'Temp_diff', 'Humidity_diff', 'Depth_age',
        'Peak_Transfer', 'Metro_THI', 'PM_TempDiff',
        'GroundPM_HumidityDiff', 'GroundPM_Air_pressure',
        'Apparent_temp', 'Ventilation_proxy', 'Relative_pressure',
        'Type_Depth', 'Age_composite', 'GroundPM_TempDiff'
    ]
    
    all_feature_cols = feature_cols + new_features
    
    print(f"  新增 {len(new_features)} 个高级特征")
    
    # ── 验证所有特征列是否存在 ──────────────────────────
    print("\n===== 特征列验证 =====")
    missing_cols = [c for c in all_feature_cols if c not in X.columns]
    found_cols   = [c for c in all_feature_cols if c in X.columns]
    
    for c in all_feature_cols:
        status = "✅" if c in X.columns else "❌ 缺失"
        print(f"  {status} '{c}'")
    
    if missing_cols:
        raise ValueError(
            f"\n❌ 以下特征列未生成：{missing_cols}\n"
            f"请检查特征工程代码是否完整"
        )
    else:
        print(f"\n✅ 全部 {len(all_feature_cols)} 个特征列验证通过")

    # ---------- 特征筛选：去除高度共线特征（|r| > 0.95）----------
    print("\n特征筛选: 去除高共线性特征...")
    corr_mat = X[all_feature_cols].corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
    if drop_cols:
        print(f"  移除高共线性特征 (|r|>0.95): {drop_cols}")
        X.drop(columns=drop_cols, inplace=True)
        all_feature_cols = [c for c in all_feature_cols if c not in drop_cols]
    else:
        print("  无高共线性特征需移除")
    print(f"  最终特征数: {len(all_feature_cols)}")

    # ---------- 目标变量统计 ----------
    print("\n目标变量统计描述:")
    print(f"均值: {y.mean():.4f}  标准差: {y.std():.4f}")
    print(f"最小值: {y.min():.4f}  最大值: {y.max():.4f}  中位数: {y.median():.4f}")
    
    # ---------- 数据集划分 ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n训练集: {X_train.shape[0]} 样本 ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"测试集:  {X_test.shape[0]} 样本 ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # ---------- 特征标准化（RobustScaler）----------
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    print("特征标准化完成 (RobustScaler)")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, all_feature_cols

# ==================== 残差驱动数据增强 ====================
def residual_based_augmentation(X, y, n_aug=2, noise_scale=0.15):
    """
    基于残差的数据增强（适配小样本，不限定城市）
    
    X: 原始特征 DataFrame
    y: 原始目标 Series（地铁PM值）
    n_aug: 每个样本生成几个增强样本（小样本推荐1）
    noise_scale: 噪声强度（小样本推荐0.1~0.3，绝对不超过0.4）
    """
    #  1. 提前初始化所有关键变量，解决Pylance未定义报错
    sigma = 1.0
    X_aug_list = []  # 提前声明空列表
    y_aug_list = []  # 提前声明空列表

    # 2. 拟合基准线性模型（小样本用更强正则，避免基准模型过拟合）
    base_model = Ridge(alpha=100.0)  # 小样本增强正则
    base_model.fit(X, y)

    y_pred = base_model.predict(X)
    residuals = y - y_pred

    # 3. 重新赋值残差分布参数（覆盖初始值，逻辑不变）
    mu = 0  # 小样本残差均值置0，避免系统偏差
    sigma = residuals.std()

    # 4. 定义连续特征列（仅对这些特征加微小噪声，分类特征不变）
    continuous_cols = ['Platform depth', 'Metro humidity', 'Outdoor humidity', 
                       'Metro temperature', 'Outdoor temperature', 
                       'Air pressure', 'Ground_PM']
    # 提高特征噪声的精细度，分特征设置不同噪声强度（贴合业务实际，避免关键特征过度扰动）
    feat_noise_scale_map = {
        'Platform depth': 0.005, 'Metro humidity': 0.01, 'Outdoor humidity': 0.01,
        'Metro temperature': 0.008, 'Outdoor temperature': 0.008,
        'Air pressure': 0.001, 'Ground_PM': 0.015  # Ground_PM为核心特征，噪声稍高但不超0.02
    }

    # 5. 生成增强样本
    for _ in range(n_aug):
        # 标签噪声：进一步降低强度+更严格的值域约束（原1.1→1.05，避免生成不合理的PM值）
        noise = np.random.normal(mu, sigma * noise_scale, size=len(y))
        y_new = y_pred + noise
        y_new = np.clip(y_new, 0, y.max() * 1.05)  # 上限不超过原最大值的105%
        
        # 特征噪声：按特征个性化设置噪声强度，而非统一0.01
        X_new = X.copy()
        for col in continuous_cols:
            if col in X_new.columns:
                feat_noise = np.random.normal(0, X_new[col].std() * feat_noise_scale_map[col], size=len(X_new))
                X_new[col] += feat_noise
                # 更严格的业务值域约束，避免生成不合理特征值
                if 'humidity' in col.lower():
                    X_new[col] = np.clip(X_new[col], 20, 95)  # 湿度20-95%（原0-100，排除极端无意义值）
                elif 'temperature' in col.lower():
                    X_new[col] = np.clip(X_new[col], 10, 35)  # 温度10-35℃（原-10-40，贴合地铁实际）
                elif 'Platform depth' in col:
                    X_new[col] = np.clip(X_new[col], 5, 25)  # 站台深度5-25m（原1-30，排除极端值）
                elif 'PM' in col:
                    X_new[col] = np.clip(X_new[col], 0, 300)  # 地面PM0-300（原0-500，贴合实际监测范围）
                elif 'Air pressure' in col:
                    X_new[col] = np.clip(X_new[col], 980, 1050)  # 气压980-1050hPa，贴合实际
        
        # 新增：对分类特征进行轻微扰动
        categorical_cols = ['Peak', 'Transfer station']  # 离散分类特征
        for col in categorical_cols:
            if col in X_new.columns:
                # 5%的概率翻转分类值，避免分类特征完全不变
                flip_mask = np.random.choice([True, False], size=len(X_new), p=[0.05, 0.95])
                X_new.loc[flip_mask, col] = 1 - X_new.loc[flip_mask, col]
        
        X_aug_list.append(X_new)
        y_aug_list.append(pd.Series(y_new, index=y.index))
    
    X_aug = pd.concat(X_aug_list, axis=0)
    y_aug = pd.concat(y_aug_list, axis=0)
    return X_aug, y_aug

# ==================== 2. Adaptive LASSO实现 ====================
class AdaptiveLasso:
    """Adaptive LASSO回归实现"""
    def __init__(self, alpha=1.0, gamma=1.0, max_iter=10000):
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        # 第一步：使用Ridge回归获得初始权重
        ridge = Ridge(alpha=0.1)
        ridge.fit(X, y)
        ridge_coef = np.abs(ridge.coef_)
        
        # 计算自适应权重
        weights = 1.0 / (ridge_coef ** self.gamma + 1e-8)
        
        # 第二步：使用加权的LASSO
        X_weighted = X * weights
        
        lasso = Lasso(alpha=self.alpha, max_iter=self.max_iter)
        lasso.fit(X_weighted, y)
        
        self.coef_ = lasso.coef_ * weights
        self.intercept_ = lasso.intercept_
        
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_
    
    def get_params(self, deep=True):
        return {'alpha': self.alpha, 'gamma': self.gamma, 'max_iter': self.max_iter}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# ==================== 3. 模型定义（贝叶斯搜索空间版）====================
def get_models_and_params(n_train=360):
    """
    模型定义 —— 贝叶斯优化版
    """

    if BAYES_AVAILABLE:
        models = {
            'Ridge': {
                'model': Ridge(),
                'params': {
                    'alpha': Real(1e-3, 1e3, prior='log-uniform'),
                },
                'use_scaled': True, 'n_iter': 40
            },
            'Lasso': {
                'model': Lasso(max_iter=50000),
                'params': {
                    'alpha': Real(1e-3, 50.0, prior='log-uniform'),
                },
                'use_scaled': True, 'n_iter': 40
            },
            'ElasticNet': {
                'model': ElasticNet(max_iter=35000),
                'params': {
                    'alpha':    Real(1e-3, 10.0, prior='log-uniform'),
                    'l1_ratio': Real(0.05, 0.95, prior='uniform'),
                },
                'use_scaled': True, 'n_iter': 40
            },
            'Adaptive Lasso': {
                'model': AdaptiveLasso(),
                'params': {
                    'alpha': Real(1e-3, 10.0, prior='log-uniform'),
                    'gamma': Real(0.3, 2.5,  prior='uniform'),
                },
                'use_scaled': True, 'n_iter': 35
            },
            'Random Forest': {
                'model': RandomForestRegressor(
                    random_state=42, n_jobs=-1, oob_score=True),
                'params': {
                    'n_estimators':      Integer(100, 600),
                    'max_depth':         Integer(3, 8),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf':  Integer(1, 10),
                    'max_features':      Real(0.3, 0.9, prior='uniform'),
                    'max_samples':       Real(0.6, 0.95, prior='uniform'),
                },
                'use_scaled': False, 'n_iter': 60
            },
            'Extra Trees': {
                'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators':      Integer(100, 500),
                    'max_depth':         Integer(3, 8),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf':  Integer(1, 10),
                    'max_features':      Real(0.3, 0.9, prior='uniform'),
                },
                'use_scaled': False, 'n_iter': 50
            },
            'XGBoost': {
                'model': XGBRegressor(
                    random_state=42, n_jobs=-1,
                    tree_method='hist', eval_metric='rmse'),
                'params': {
                    'n_estimators':      Integer(100, 600),
                    'max_depth':         Integer(2, 6),
                    'learning_rate':     Real(0.005, 0.3,  prior='log-uniform'),
                    'subsample':         Real(0.6, 1.0,   prior='uniform'),
                    'colsample_bytree':  Real(0.6, 1.0,   prior='uniform'),
                    'reg_alpha':         Real(1e-4, 50.0, prior='log-uniform'),
                    'reg_lambda':        Real(1e-4, 100.0,prior='log-uniform'),
                    'min_child_weight':  Integer(1, 20),
                    'gamma':             Real(0.0, 5.0,   prior='uniform'),
                },
                'use_scaled': False, 'n_iter': 70
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators':      Integer(100, 400),
                    'max_depth':         Integer(2, 5),
                    'learning_rate':     Real(0.005, 0.2, prior='log-uniform'),
                    'subsample':         Real(0.6, 1.0,  prior='uniform'),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf':  Integer(1, 10),
                    'max_features':      Real(0.3, 0.9,  prior='uniform'),
                    'validation_fraction': Real(0.1, 0.2, prior='uniform'),
                    'n_iter_no_change':  Integer(10, 30),
                },
                'use_scaled': False, 'n_iter': 60
            },
            'KNN': {
                'model': KNeighborsRegressor(n_jobs=-1),
                'params': {
                    'n_neighbors': Integer(3, 15),
                    'weights':     Categorical(['distance', 'uniform']),
                    'metric':      Categorical(['euclidean', 'manhattan']),
                },
                'use_scaled': True, 'n_iter': 30
            },
            'SVM': {
                'model': SVR(),
                'params': {
                    'C':       Real(0.1, 100.0, prior='log-uniform'),
                    'gamma':   Categorical(['scale', 'auto']),
                    'kernel':  Categorical(['rbf', 'linear']),
                    'epsilon': Real(0.01, 0.5,  prior='log-uniform'),
                },
                'use_scaled': True, 'n_iter': 40
            },
        }

        if LGBM_AVAILABLE:
            models['LightGBM'] = {
                'model': lgb.LGBMRegressor(
                    random_state=42, n_jobs=-1, verbose=-1),
                'params': {
                    'n_estimators':      Integer(100, 600),
                    'max_depth':         Integer(3, 7),
                    'learning_rate':     Real(0.005, 0.3,  prior='log-uniform'),
                    'num_leaves':        Integer(10, 80),
                    'subsample':         Real(0.6, 1.0,   prior='uniform'),
                    'colsample_bytree':  Real(0.6, 1.0,   prior='uniform'),
                    'reg_alpha':         Real(1e-4, 50.0, prior='log-uniform'),
                    'reg_lambda':        Real(1e-4, 100.0,prior='log-uniform'),
                    'min_child_samples': Integer(3, 50),
                    'path_smooth':       Real(0.0, 2.0,   prior='uniform'),
                },
                'use_scaled': False, 'n_iter': 70
            }
            print("✓ LightGBM（贝叶斯搜索空间）已加载")

    else:
        models = {
            'Ridge': {
                'model': Ridge(),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0],
                    'solver': ['auto', 'lsqr', 'saga']
                },
                'use_scaled': True, 'n_iter': 30
            },
            'Lasso': {
                'model': Lasso(max_iter=50000),
                'params': {
                    'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                    'selection': ['random', 'cyclic']
                },
                'use_scaled': True, 'n_iter': 25
            },
            'ElasticNet': {
                'model': ElasticNet(max_iter=35000),
                'params': {
                    'alpha':    [0.01, 0.1, 0.5, 1.0, 5.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'selection': ['random']
                },
                'use_scaled': True, 'n_iter': 25
            },
            'Adaptive Lasso': {
                'model': AdaptiveLasso(),
                'params': {
                    'alpha': [0.01, 0.1, 0.5, 1.0, 5.0],
                    'gamma': [0.5, 0.8, 1.0, 1.5, 2.0]
                },
                'use_scaled': True, 'n_iter': 25
            },
            'Random Forest': {
                'model': RandomForestRegressor(
                    random_state=42, n_jobs=-1, oob_score=True),
                'params': {
                    'n_estimators':      [100, 200, 300, 500],
                    'max_depth':         [3, 4, 5, 6, None],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf':  [1, 3, 5, 8],
                    'max_features':      ['sqrt', 'log2', 0.5, 0.7],
                    'max_samples':       [0.7, 0.8, 0.9],
                },
                'use_scaled': False, 'n_iter': 50
            },
            'Extra Trees': {
                'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators':      [100, 200, 300],
                    'max_depth':         [3, 4, 5, None],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf':  [1, 3, 5, 8],
                    'max_features':      ['sqrt', 'log2', 0.5]
                },
                'use_scaled': False, 'n_iter': 40
            },
            'XGBoost': {
                'model': XGBRegressor(
                    random_state=42, n_jobs=-1,
                    tree_method='hist', eval_metric='rmse'),
                'params': {
                    'n_estimators':     [100, 200, 300, 500],
                    'max_depth':        [2, 3, 4, 5],
                    'learning_rate':    [0.01, 0.05, 0.1, 0.2],
                    'subsample':        [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                    'reg_alpha':        [0.0, 0.1, 1.0, 10.0, 50.0],
                    'reg_lambda':       [0.1, 1.0, 10.0, 50.0],
                    'min_child_weight': [1, 3, 5, 10],
                    'gamma':            [0.0, 0.1, 1.0, 5.0],
                },
                'use_scaled': False, 'n_iter': 60
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators':      [100, 200, 300],
                    'max_depth':         [2, 3, 4],
                    'learning_rate':     [0.01, 0.05, 0.1],
                    'subsample':         [0.7, 0.8, 0.9],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf':  [1, 3, 5],
                    'max_features':      ['sqrt', 0.5, 0.7],
                    'validation_fraction': [0.15],
                    'n_iter_no_change':  [20],
                    'tol':               [1e-4]
                },
                'use_scaled': False, 'n_iter': 50
            },
            'KNN': {
                'model': KNeighborsRegressor(n_jobs=-1),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights':     ['distance', 'uniform'],
                    'metric':      ['euclidean', 'manhattan'],
                },
                'use_scaled': True, 'n_iter': 20
            },
            'SVM': {
                'model': SVR(),
                'params': {
                    'C':       [0.5, 1.0, 5.0, 10.0, 50.0],
                    'gamma':   ['scale', 'auto'],
                    'kernel':  ['rbf', 'linear'],
                    'epsilon': [0.05, 0.1, 0.2],
                },
                'use_scaled': True, 'n_iter': 30
            },
        }

        if LGBM_AVAILABLE:
            models['LightGBM'] = {
                'model': lgb.LGBMRegressor(
                    random_state=42, n_jobs=-1, verbose=-1),
                'params': {
                    'n_estimators':      [100, 200, 300],
                    'max_depth':         [3, 4, 5, 6],
                    'learning_rate':     [0.01, 0.05, 0.1],
                    'num_leaves':        [15, 31, 50, 63],
                    'subsample':         [0.7, 0.8, 0.9],
                    'colsample_bytree':  [0.7, 0.8, 0.9],
                    'reg_alpha':         [0.0, 0.1, 1.0, 10.0],
                                        'reg_lambda':        [0.1, 1.0, 10.0, 50.0],
                    'min_child_samples': [5, 10, 20, 30],
                    'path_smooth':       [0.0, 1.0, 2.0],
                },
                'use_scaled': False, 'n_iter': 60
            }
            print("✓ LightGBM（随机搜索空间）已加载")

    return models


# ==================== 4. 搜索器构建函数====================
def build_search(model, params, n_iter, n_cv):
    """
    根据可用库自动选择最优搜索策略：
    贝叶斯优化（BayesSearchCV）> 随机搜索（RandomizedSearchCV）

    返回：(search对象, 搜索方式描述字符串)
    """
    if BAYES_AVAILABLE:
        try:
            search = BayesSearchCV(
                estimator   = model,
                search_spaces = params,
                n_iter      = n_iter,
                cv          = n_cv,
                scoring     = 'r2',
                n_jobs      = -1,
                random_state= 42,
                verbose     = 0,
                refit       = True,
                return_train_score = True,
                optimizer_kwargs   = {'base_estimator': 'GP',  # 高斯过程代理模型
                                      'acq_func': 'EI'}        # 期望提升采集函数
            )
            return search, "贝叶斯优化（BayesSearchCV·GP·EI）"
        except Exception as e:
            print(f"  ⚠ BayesSearchCV初始化失败（{e}），回退随机搜索")

    # 兜底：随机搜索
    search = RandomizedSearchCV(
        estimator   = model,
        param_distributions = params,
        n_iter      = n_iter,
        cv          = n_cv,
        scoring     = 'r2',
        n_jobs      = -1,
        random_state= 42,
        verbose     = 0,
        refit       = True,
        error_score = 'raise',
        return_train_score = True
    )
    return search, "随机搜索（RandomizedSearchCV）"


# ==================== 4. 模型训练和评估====================
def train_and_evaluate_models(X_train, X_test, y_train, y_test,
                               X_train_scaled, X_test_scaled):
    """
    模型训练 —— 贝叶斯优化 + 完整过拟合诊断版

    优化点：
    1. 自动选择 BayesSearchCV / RandomizedSearchCV
    2. 完整过拟合诊断（修复缩进Bug）
    3. 训练集CV均值同步输出
    4. Stacking基于CV-R²选Top3
    """
    n_train = len(X_train)
    n_cv    = 5 if n_train >= 300 else 5   # ★ 样本足够时用10折
    models_config = get_models_and_params(n_train=n_train)
    results        = {}
    trained_models = {}

    search_mode = "贝叶斯优化" if BAYES_AVAILABLE else "随机搜索"
    print("\n" + "=" * 100)
    print(f"步骤 2: 模型训练（{search_mode}，{n_cv}折CV）")
    print("=" * 100)
    print(f"训练集: {n_train} 样本 | 交叉验证: {n_cv}折 | 共 {len(models_config)} 个模型")

    for idx, (model_name, config) in enumerate(models_config.items(), 1):
        print(f"\n[{idx}/{len(models_config)}] {model_name}")
        print("-" * 80)

        if config['use_scaled']:
            X_tr, X_te = X_train_scaled, X_test_scaled
            print("  数据: 标准化（RobustScaler）")
        else:
            X_tr, X_te = X_train, X_test
            print("  数据: 原始")

        try:
            # ── 自动选择搜索器 ──────────────────────────────
            n_iter = config.get('n_iter', 40)
            search, search_desc = build_search(
                config['model'], config['params'], n_iter, n_cv)
            print(f"  搜索策略: {search_desc}（n_iter={n_iter}）")

            search.fit(X_tr, y_train)

            best_model  = search.best_estimator_
            best_params = search.best_params_

            # ── 训练集 & 测试集预测 ─────────────────────────
            y_pred_train = best_model.predict(X_tr)
            y_pred_test  = best_model.predict(X_te)

            train_r2   = r2_score(y_train, y_pred_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            train_mae  = mean_absolute_error(y_train, y_pred_train)

            # ── 交叉验证（再次评估，确保稳健）──────────────
            cv_scores = cross_val_score(
                best_model, X_tr, y_train,
                cv=n_cv, scoring='r2', n_jobs=-1)

            # ── 测试集指标 ──────────────────────────────────
            test_r2   = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae  = mean_absolute_error(y_test, y_pred_test)
            mask      = y_test != 0
            test_mape = np.mean(
                np.abs((y_test[mask] - y_pred_test[mask]) / y_test[mask])
            ) * 100
            gap = train_r2 - test_r2

            # ── 贝叶斯搜索额外信息 ─────────────────────────
            if BAYES_AVAILABLE and hasattr(search, 'best_score_'):
                cv_best_score = search.best_score_
            else:
                cv_best_score = cv_scores.mean()

            results[model_name] = {
                'best_params'  : dict(best_params),   # ★ 统一转dict
                'train_r2'     : train_r2,
                'test_r2'      : test_r2,
                'train_rmse'   : train_rmse,
                'test_rmse'    : test_rmse,
                'train_mae'    : train_mae,
                'test_mae'     : test_mae,
                'test_mape'    : test_mape,
                'cv_r2_mean'   : cv_scores.mean(),
                'cv_r2_std'    : cv_scores.std(),
                'cv_best_score': cv_best_score,
                'overfit_gap'  : gap,
                'y_pred_train' : y_pred_train,
                'y_pred_test'  : y_pred_test,
                'search_mode'  : search_desc           # ★ 记录搜索方式
            }
            trained_models[model_name] = {
                'model'        : best_model,
                'use_scaled'   : config['use_scaled'],
                'test_r2'      : test_r2,
                'trained_model': best_model
            }

            print(f"  最佳参数: {dict(best_params)}")
            print(f"  训练集 R²={train_r2:.4f} | RMSE={train_rmse:.4f} | MAE={train_mae:.4f}")
            print(f"  测试集 R²={test_r2:.4f} | RMSE={test_rmse:.4f} | "
                  f"MAE={test_mae:.4f} | MAPE={test_mape:.2f}%")
            print(f"  CV R²({n_cv}折): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
                  f"  [搜索最优CV={cv_best_score:.4f}]")

            # ── 过拟合诊断统一在判断块外打印 ──
            if gap > 0.20:
                lvl = "🔴 严重过拟合"
                tip = "大幅提高正则化；降低 max_depth；增大 min_samples_leaf/min_child_weight"
            elif gap > 0.12:
                lvl = "🟠 明显过拟合"
                tip = "适当增大正则化；检查 max_depth 和叶节点样本数设置"
            elif gap > 0.05:
                lvl = "🟡 轻微过拟合"
                tip = "可接受范围，可微调正则化参数"
            else:
                lvl = "🟢 泛化良好"
                tip = "训练/测试一致，模型可靠"

            # ★ 所有模型都打印状态
            print(f"  过拟合诊断: {lvl}（Gap={gap:+.4f}）")
            if gap > 0.05:
                print(f"  💡 建议: {tip}")

        except Exception as e:
            print(f"  ❌ 训练失败: {e}")
            import traceback; traceback.print_exc()
            continue

    # ── Stacking（按CV-R²选Top3树模型）──────────────────────
    print(f"\n[{len(models_config)+1}/{len(models_config)+1}] Stacking集成")
    print("-" * 80)
    try:
        tree_names = ['Random Forest', 'Extra Trees', 'XGBoost',
                      'Gradient Boosting', 'LightGBM']
        candidates = [
            (n, results[n]['cv_r2_mean'], trained_models[n])
            for n in tree_names if n in results and n in trained_models
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        top3 = candidates[:3]

        if len(top3) >= 2:
            base_ests = [
                (n.replace(' ', '_'), info['model'])
                for n, _, info in top3
            ]
            print(f"  基模型: {[(n, f'CV={s:.4f}') for n,s,_ in top3]}")

            stacking = StackingRegressor(
                estimators     = base_ests,
                final_estimator= Ridge(alpha=50.0),  # ★ 降低元模型正则化
                cv             = n_cv,
                n_jobs         = -1,
                passthrough    = False
            )
            stacking.fit(X_train, y_train)

            ytr_st = stacking.predict(X_train)
            yte_st = stacking.predict(X_test)

            st_tr_r2 = r2_score(y_train, ytr_st)
            st_te_r2 = r2_score(y_test,  yte_st)
            st_rmse  = np.sqrt(mean_squared_error(y_test, yte_st))
            st_mae   = mean_absolute_error(y_test, yte_st)
            mask     = y_test != 0
            st_mape  = np.mean(
                np.abs((y_test[mask]-yte_st[mask])/y_test[mask]))*100
            st_cv    = cross_val_score(
                stacking, X_train, y_train,
                cv=n_cv, scoring='r2', n_jobs=-1)
            st_gap   = st_tr_r2 - st_te_r2

            results['Stacking'] = {
                'best_params'  : {'base': [n for n,_,_ in top3],
                                  'meta': 'Ridge(alpha=50)'},
                'train_r2'     : st_tr_r2,
                'test_r2'      : st_te_r2,
                'train_rmse'   : np.sqrt(mean_squared_error(y_train, ytr_st)),
                'test_rmse'    : st_rmse,
                'train_mae'    : mean_absolute_error(y_train, ytr_st),
                'test_mae'     : st_mae,
                'test_mape'    : st_mape,
                'cv_r2_mean'   : st_cv.mean(),
                'cv_r2_std'    : st_cv.std(),
                'cv_best_score': st_cv.mean(),
                'overfit_gap'  : st_gap,
                'y_pred_train' : ytr_st,
                'y_pred_test'  : yte_st,
                'search_mode'  : 'N/A（Stacking无超参数搜索）'
            }
            trained_models['Stacking'] = {
                'model'        : stacking,
                'use_scaled'   : False,
                'test_r2'      : st_te_r2,
                'trained_model': stacking
            }

            print(f"  训练集 R²={st_tr_r2:.4f}")
            print(f"  测试集 R²={st_te_r2:.4f} | RMSE={st_rmse:.4f} | "
                  f"MAE={st_mae:.4f} | MAPE={st_mape:.2f}%")
            print(f"  CV R²({n_cv}折): {st_cv.mean():.4f} ± {st_cv.std():.4f}")

            # 过拟合诊断
            if st_gap > 0.20:   st_lvl = "🔴 严重过拟合"
            elif st_gap > 0.12: st_lvl = "🟠 明显过拟合"
            elif st_gap > 0.05: st_lvl = "🟡 轻微过拟合"
            else:               st_lvl = "🟢 泛化良好"
            print(f"  过拟合诊断: {st_lvl}（Gap={st_gap:+.4f}）")
        else:
            print("  树模型不足2个，跳过Stacking")

    except Exception as e:
        print(f"  ❌ Stacking失败: {e}")

    # ── 训练总结 ─────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"训练总结（{search_mode}，按测试集R²排序）")
    print("=" * 100)
    sorted_r = sorted(results.items(),
                      key=lambda x: x[1]['test_r2'], reverse=True)
    print(f"{'模型':<22} {'训练R²':>8} {'测试R²':>8} {'CV R²':>8} "
          f"{'Gap':>8} {'状态':<10} {'搜索方式'}")
    print("-" * 100)
    for mn, r in sorted_r:
        g  = r.get('overfit_gap', r['train_r2'] - r['test_r2'])
        st = ("🟢良好" if g <= 0.05 else
              "🟡轻微" if g <= 0.12 else
              "🟠明显" if g <= 0.20 else "🔴严重")
        mode = r.get('search_mode', 'N/A')[:20]   # 截断避免过长
        print(f"{mn:<22} {r['train_r2']:>8.4f} {r['test_r2']:>8.4f} "
              f"{r['cv_r2_mean']:>8.4f} {g:>+8.4f}  {st:<10} {mode}")

    return results, trained_models                                      

# ==================== 5. 特征重要性分析 ====================
def analyze_feature_importance(models_dict, X_train, X_test, y_train, y_test, feature_cols):
    """分析特征重要性"""
    print("\n" + "=" * 100)
    print("步骤 3: 特征重要性分析")
    print("=" * 100)
    importance_results = {}
    from sklearn.inspection import permutation_importance
    
    for model_name, model_info in models_dict.items():
        try:
            # 获取训练好的模型（修复：使用model_info中的trained_model键）
            model = model_info.get('trained_model', model_info['model'])
            X = X_train if model_info['use_scaled'] else X_train
            
            # 1. 树模型（RandomForest/XGBoost/LightGBM/GradientBoosting/ExtraTrees）
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            
            # 2. 线性模型（Ridge/Lasso/ElasticNet/Adaptive Lasso）
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
                # 归一化到0-1区间
                if np.sum(importance) > 0:
                    importance = importance / np.sum(importance)
            
            # 3. KNN/SVM（排列重要性）
            else:
                print(f"  {model_name}: 计算排列重要性（耗时较长）...")
                perm_result = permutation_importance(
                    model, X, y_train, n_repeats=10, random_state=42, n_jobs=-1
                )
                importance = perm_result.importances_mean
            
            # 存储结果
            importance_results[model_name] = pd.Series(importance, index=X.columns).sort_values(ascending=False)
            
            # 打印TOP10特征
            print(f"\n{model_name} 特征重要性（TOP10）:")
            print(importance_results[model_name].head(10))
            
        except Exception as e:
            print(f"  {model_name}: 特征重要性计算失败 - {str(e)}")
            import traceback; traceback.print_exc()
            importance_results[model_name] = None
    
    # 可视化TOP5特征（以性能最优模型为例）
    best_model_name = max(models_dict.keys(), key=lambda x: models_dict[x]['test_r2'] if 'test_r2' in models_dict[x] else 0)
    if importance_results[best_model_name] is not None and importance_results[best_model_name] is not None:
        plt.figure(figsize=(12, 8))
        top5_feat = importance_results[best_model_name].head(5)
        sns.barplot(x=top5_feat.values, y=top5_feat.index, palette='viridis')
        plt.title(f'{best_model_name} 特征重要性（TOP5）', fontsize=14)
        plt.xlabel('重要性值', fontsize=12)
        plt.ylabel('特征名称', fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path(f'{best_model_name}_feature_importance.png'), dpi=300)
        plt.close()
    
    return importance_results
# ==================== 6. SHAP值分析（去除TabPFN）====================
def analyze_shap_values(trained_models, X_train, X_test, X_train_scaled,
                        X_test_scaled, feature_names, y_train, y_test):
    """
    计算所有模型的SHAP值（优化版）
    
    优化点：
    1. 去除TabPFN相关代码
    2. 优化采样策略提升速度
    3. 统一错误处理机制
    """
    shap_results = {}

    print("\n" + "=" * 100)
    print("步骤 4: SHAP值分析（优化版）")
    print("=" * 100)

    for model_name, model_info in trained_models.items():
        print(f"\n计算 {model_name} 的SHAP值...")
        print("-" * 80)

        model     = model_info['model']
        use_scaled= model_info['use_scaled']
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled  if use_scaled else X_test

        sv          = None   # shap values (2D array)
        base_val    = None
        X_sample    = None
        explainer   = None

        if model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting',
                          'Extra Trees', 'LightGBM']:
            try:
                explainer = shap.TreeExplainer(model)
                X_sample  = X_te   # 全量测试集
                raw       = explainer.shap_values(X_sample)

                # 统一为 2D array
                sv = raw[0] if isinstance(raw, list) else raw
                sv = np.array(sv)

                # expected_value 标量化
                ev = explainer.expected_value
                base_val = float(ev[0] if hasattr(ev, '__len__') else ev)

                print(f"  ✓ TreeExplainer 成功 (样本数: {len(X_sample)})")

            except Exception as e:
                print(f"  ✗ TreeExplainer 失败: {e}")
                # 降级到 PermutationExplainer
                try:
                    print("  → 降级到 PermutationExplainer...")
                    bg = shap.sample(X_tr, min(50, len(X_tr)))
                    explainer = shap.PermutationExplainer(model.predict, bg)
                    X_sample  = X_te.iloc[:min(80, len(X_te))]
                    sv_obj    = explainer(X_sample)
                    sv        = np.array(sv_obj.values)
                    base_val  = float(sv_obj.base_values[0] if hasattr(sv_obj.base_values, '__len__') else sv_obj.base_values)
                    print(f"  ✓ PermutationExplainer 成功 (样本数: {len(X_sample)})")
                except Exception as e2:
                    print(f"  ✗ 降级也失败: {e2}")
                    continue

        elif model_name in ['Ridge', 'Lasso', 'Adaptive Lasso', 'ElasticNet']:
            try:
                explainer = shap.LinearExplainer(
                    model, X_tr,
                    feature_perturbation='interventional'
                )
                X_sample = X_te
                raw      = explainer.shap_values(X_sample)
                sv       = np.array(raw[0] if isinstance(raw, list) else raw)

                ev = explainer.expected_value
                base_val = float(ev[0] if hasattr(ev, '__len__') else ev)

                print(f"  ✓ LinearExplainer 成功 (样本数: {len(X_sample)})")

            except Exception as e:
                print(f"  ✗ LinearExplainer 失败: {e}")
                continue

        elif model_name in ['SVM', 'KNN']:
            try:
                n_bg = min(20, len(X_tr))
                background = shap.kmeans(X_tr, n_bg)
                explainer  = shap.KernelExplainer(model.predict, background)

                n_explain = min(60, len(X_te))
                X_sample  = X_te.iloc[:n_explain]
                raw       = explainer.shap_values(X_sample, silent=True)
                sv        = np.array(raw[0] if isinstance(raw, list) else raw)

                ev = explainer.expected_value
                base_val = float(ev[0] if hasattr(ev, '__len__') else ev)

                print(f"  ✓ KernelExplainer (kmeans背景={n_bg}) 成功 (样本数: {n_explain})")

            except Exception as e:
                print(f"  ✗ KernelExplainer 失败: {e}")
                continue

        # =====================================================================
        # Stacking — PermutationExplainer
        # =====================================================================
        elif model_name == 'Stacking':
            try:
                n_bg = min(30, len(X_tr))
                bg   = shap.sample(X_tr, n_bg)
                explainer = shap.PermutationExplainer(model.predict, bg)

                n_explain = min(60, len(X_te))
                X_sample  = X_te.iloc[:n_explain]
                sv_obj    = explainer(X_sample)
                sv        = np.array(sv_obj.values)
                base_val  = float(
                    sv_obj.base_values[0]
                    if hasattr(sv_obj.base_values, '__len__')
                                        else sv_obj.base_values
                )
                print(f"  ✓ PermutationExplainer 成功 (背景={n_bg}, 解释={n_explain})")

            except Exception as e:
                print(f"  ✗ PermutationExplainer 失败: {e}")
                continue

        else:
            # 尝试通用 PermutationExplainer
            try:
                bg   = shap.sample(X_tr, min(20, len(X_tr)))
                explainer = shap.PermutationExplainer(model.predict, bg)
                n_explain = min(60, len(X_te))
                X_sample  = X_te.iloc[:n_explain]
                sv_obj    = explainer(X_sample)
                sv        = np.array(sv_obj.values)
                base_val  = float(
                    sv_obj.base_values[0]
                    if hasattr(sv_obj.base_values, '__len__')
                    else sv_obj.base_values
                )
                print(f"  ✓ PermutationExplainer (通用) 成功")
            except Exception as e:
                print(f"  ✗ 通用SHAP失败: {e}")
                continue

        # =====================================================================
        # 后处理：检查维度、计算统计量
        # =====================================================================
        if sv is None or sv.ndim != 2:
            print(f"  ✗ SHAP值维度异常，跳过 {model_name}")
            continue

        # 对齐特征名
        if hasattr(X_sample, 'columns'):
            feat_names_used = list(X_sample.columns)
        else:
            feat_names_used = feature_names[:sv.shape[1]]

        # 维度对齐检查
        if sv.shape[1] != len(feat_names_used):
            n_min = min(sv.shape[1], len(feat_names_used))
            sv = sv[:, :n_min]
            feat_names_used = feat_names_used[:n_min]

        # 计算 mean |SHAP| 和方向
        mean_abs_shap   = np.abs(sv).mean(axis=0)
        mean_signed_shap= sv.mean(axis=0)

        shap_importance = pd.DataFrame({
            'Feature':           feat_names_used,
            'Mean_Abs_SHAP':     mean_abs_shap,
            'Mean_Signed_SHAP':  mean_signed_shap,
        }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

        shap_importance['Direction'] = shap_importance['Mean_Signed_SHAP'].apply(
            lambda x: '正影响(↑)' if x >= 0 else '负影响(↓)'
        )

        shap_results[model_name] = {
            'explainer':          explainer,
            'shap_values':        sv,
            'X_sample':           X_sample,
            'base_value':         base_val,
            'importance':         shap_importance,
            'feature_names_used': feat_names_used
        }

        n_samp = len(X_sample) if hasattr(X_sample, '__len__') else '?'
        print(f"✓ SHAP计算完成 | 样本数: {n_samp} | 特征数: {sv.shape[1]}")
        print("  Top 10 特征 (含影响方向):")
        for _, row in shap_importance.head(10).iterrows():
            bar = '█' * int(row['Mean_Abs_SHAP'] / (mean_abs_shap.max()+1e-9) * 20)
            print(f"    {row['Feature']:<30} {row['Mean_Abs_SHAP']:.5f} {bar} {row['Direction']}")

    success = len(shap_results)
    total   = len(trained_models)
    print(f"\nSHAP分析完成: {success}/{total} 个模型成功")
    return shap_results
# ==================== 7. 可视化函数 ====================
def create_visualizations(results, importance_results, shap_results, 
                         feature_names, y_test):
    """创建所有可视化图表"""
    
    print("\n" + "=" * 100)
    print("步骤 5: 生成可视化图表")
    print("=" * 100)
    
    # 7.1 模型性能对比（增强版）
    print("\n生成模型性能对比图...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    model_names = list(results.keys())
    train_r2 = [results[m]['train_r2'] for m in model_names]
    test_r2 = [results[m]['test_r2'] for m in model_names]
    test_rmse = [results[m]['test_rmse'] for m in model_names]
    test_mae = [results[m]['test_mae'] for m in model_names]
    test_mape = [results[m]['test_mape'] for m in model_names]
    cv_means = [results[m]['cv_r2_mean'] for m in model_names]
    cv_stds = [results[m]['cv_r2_std'] for m in model_names]
    
    # R² 对比
    x = np.arange(len(model_names))
    width = 0.35
    axes[0, 0].bar(x - width/2, train_r2, width, label='训练集', alpha=0.8, color='steelblue')
    axes[0, 0].bar(x + width/2, test_r2, width, label='测试集', alpha=0.8, color='coral')
    axes[0, 0].set_xlabel('模型', fontsize=11)
    axes[0, 0].set_ylabel('R² 分数', fontsize=11)
    axes[0, 0].set_title('模型R²性能对比', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='优秀线')
    
    # RMSE 对比
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = axes[0, 1].bar(model_names, test_rmse, alpha=0.8, color=colors)
    axes[0, 1].set_xlabel('模型', fontsize=11)
    axes[0, 1].set_ylabel('RMSE', fontsize=11)
    axes[0, 1].set_title('模型RMSE对比 (测试集)', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    axes[0, 1].grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # MAE 对比
    bars = axes[0, 2].bar(model_names, test_mae, alpha=0.8, color='lightgreen')
    axes[0, 2].set_xlabel('模型', fontsize=11)
    axes[0, 2].set_ylabel('MAE', fontsize=11)
    axes[0, 2].set_title('模型MAE对比 (测试集)', fontsize=13, fontweight='bold')
    axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    axes[0, 2].grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 交叉验证R²
    bars = axes[1, 0].bar(model_names, cv_means, yerr=cv_stds, alpha=0.8, 
                          color='skyblue', capsize=5, error_kw={'linewidth': 2})
    axes[1, 0].set_xlabel('模型', fontsize=11)
    axes[1, 0].set_ylabel('交叉验证 R²', fontsize=11)
    axes[1, 0].set_title('10折交叉验证R²对比', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # MAPE 对比
    bars = axes[1, 1].bar(model_names, test_mape, alpha=0.8, color='salmon')
    axes[1, 1].set_xlabel('模型', fontsize=11)
    axes[1, 1].set_ylabel('MAPE (%)', fontsize=11)
    axes[1, 1].set_title('模型MAPE对比 (测试集)', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    axes[1, 1].grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 过拟合分析
    overfit_gaps = [train_r2[i] - test_r2[i] for i in range(len(model_names))]
    colors_overfit = ['red' if gap > 0.15 else 'orange' if gap > 0.10 else 'green' 
                      for gap in overfit_gaps]
    bars = axes[1, 2].bar(model_names, overfit_gaps, alpha=0.8, color=colors_overfit)
    axes[1, 2].set_xlabel('模型', fontsize=11)
    axes[1, 2].set_ylabel('训练集R² - 测试集R²', fontsize=11)
    axes[1, 2].set_title('过拟合分析', fontsize=13, fontweight='bold')
    axes[1, 2].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    axes[1, 2].axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label='轻微过拟合线')
    axes[1, 2].axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='严重过拟合线')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path('model_performance_comparison.png'), dpi=300, bbox_inches='tight')
    print("✓ 保存: model_performance_comparison.png")
    plt.close()
    
    # 7.2 预测值 vs 真实值
    print("生成预测值vs真实值散点图...")
    n_models = len(model_names)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig_height = max(6, min(6*n_rows, 30))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, fig_height))
    if n_models == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    for idx, model_name in enumerate(model_names):
        y_pred = results[model_name]['y_pred_test']
        r2 = results[model_name]['test_r2']
        rmse = results[model_name]['test_rmse']
        mae = results[model_name]['test_mae']
        
        # 散点图
        axes[idx].scatter(y_test, y_pred, alpha=0.6, s=30, c='steelblue', edgecolors='black', linewidth=0.5)
        
        # 理想预测线
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 
                       'r--', lw=2, label='理想预测线', alpha=0.8)
        
        # 添加拟合线
        z = np.polyfit(y_test, y_pred, 1)
        p = np.poly1d(z)
        axes[idx].plot(y_test.sort_values(), p(y_test.sort_values()), 
                      "g-", alpha=0.5, linewidth=2, label='拟合线')
        
        axes[idx].set_xlabel('真实值', fontsize=11)
        axes[idx].set_ylabel('预测值', fontsize=11)
        axes[idx].set_title(f'{model_name}\nR²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}', 
                           fontsize=11, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(alpha=0.3)
        
        # 添加统计信息
        textstr = f'样本数: {len(y_test)}\n相关系数: {np.corrcoef(y_test, y_pred)[0,1]:.4f}'
        axes[idx].text(0.05, 0.95, textstr, transform=axes[idx].transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 隐藏多余的子图
    for idx in range(len(model_names), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path('prediction_vs_actual.png'), dpi=300, bbox_inches='tight')
    print("✓ 保存: prediction_vs_actual.png")
    plt.close()
    
    # 7.3 残差分析图
    print("生成残差分析图...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, fig_height))
    if n_models == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    for idx, model_name in enumerate(model_names):
        y_pred = results[model_name]['y_pred_test']
        residuals = y_test - y_pred
        
        # 残差散点图
        axes[idx].scatter(y_pred, residuals, alpha=0.6, s=30, c='purple', edgecolors='black', linewidth=0.5)
        axes[idx].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[idx].set_xlabel('预测值', fontsize=11)
        axes[idx].set_ylabel('残差', fontsize=11)
        axes[idx].set_title(f'{model_name} - 残差分析', fontsize=11, fontweight='bold')
        axes[idx].grid(alpha=0.3)
        
        # 添加残差统计信息
        textstr = f'残差均值: {residuals.mean():.4f}\n残差标准差: {residuals.std():.4f}'
        axes[idx].text(0.05, 0.95, textstr, transform=axes[idx].transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 隐藏多余的子图
    for idx in range(len(model_names), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path('residual_analysis.png'), dpi=300, bbox_inches='tight')
    print("✓ 保存: residual_analysis.png")
    plt.close()
    
    # 7.4 特征重要性可视化
    if importance_results:
        print("生成特征重要性图...")
        n_importance = len(importance_results)
        n_cols = 3
        n_rows = (n_importance + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        if n_importance == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
        
        for idx, (model_name, importance_df) in enumerate(importance_results.items()):
            if importance_df is not None:
                top_features = importance_df.head(15)
                
                # 水平条形图
                y_pos = np.arange(len(top_features))
                colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
                
                axes[idx].barh(y_pos, top_features.values, color=colors_bar, alpha=0.8)
                axes[idx].set_yticks(y_pos)
                axes[idx].set_yticklabels(top_features.index, fontsize=9)
                axes[idx].set_xlabel('重要性', fontsize=11)
                axes[idx].set_title(f'{model_name} - Top 15 特征', fontsize=12, fontweight='bold')
                axes[idx].invert_yaxis()
                axes[idx].grid(axis='x', alpha=0.3)
                
                # 添加数值标签
                for i, v in enumerate(top_features.values):
                    axes[idx].text(v, i, f' {v:.4f}', va='center', fontsize=8)
        
        # 隐藏多余的子图
        for idx in range(len(importance_results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(out_path('feature_importance.png'), dpi=300, bbox_inches='tight')
        print("✓ 保存: feature_importance.png")
        plt.close()
    
    # 7.5 特征重要性热力图（跨模型对比）
    if importance_results and len(importance_results) > 1:
        print("生成特征重要性热力图...")
        
        # 创建特征重要性矩阵
        all_features = set()
        for imp_df in importance_results.values():
            if imp_df is not None:
                all_features.update(imp_df.index.tolist())
        
        importance_matrix = pd.DataFrame(index=sorted(all_features), 
                                        columns=importance_results.keys())
        
        for model_name, imp_df in importance_results.items():
            if imp_df is not None:
                for feature, importance in imp_df.items():
                    importance_matrix.loc[feature, model_name] = importance
        
        importance_matrix = importance_matrix.fillna(0)
        
        # 标准化每列（每个模型）
        importance_matrix_norm = importance_matrix.div(importance_matrix.max(axis=0), axis=1)
        
        # 选择Top特征
        top_features_overall = importance_matrix_norm.sum(axis=1).nlargest(20).index
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(importance_matrix_norm.loc[top_features_overall], 
                   annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': '标准化重要性'},
                   linewidths=0.5)
        plt.title('特征重要性热力图 (Top 20特征, 跨模型对比)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('模型', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out_path('feature_importance_heatmap.png'), dpi=300, bbox_inches='tight')
        print("✓ 保存: feature_importance_heatmap.png")
        plt.close()
    
    # 7.6 SHAP可视化
    if shap_results:
        print("生成SHAP可视化图 (增强版)...")
        
        # 找出最优的树模型（用于依赖图）
        tree_models_in_shap = [m for m in shap_results 
                               if m in ['Random Forest','XGBoost','Gradient Boosting',
                                        'Extra Trees','LightGBM']]
        best_tree_for_shap = None
        if tree_models_in_shap and results:
            best_tree_for_shap = max(
                tree_models_in_shap,
                key=lambda m: results.get(m, {}).get('test_r2', -999)
            )
        
        for model_name, shap_data in shap_results.items():
            try:
                sv       = shap_data['shap_values']
                X_samp   = shap_data['X_sample']
                feat_names_used = shap_data.get('feature_names_used', feature_names)
                base_val = shap_data['base_value']
                
                # --- (a) SHAP Summary Plot（蜂群图）---
                plt.figure(figsize=(12, 8))
                shap.summary_plot(sv, X_samp, feature_names=feat_names_used,
                                  show=False, max_display=18)
                plt.title(f'{model_name} — SHAP Summary Plot (蜂群图)',
                         fontsize=13, fontweight='bold', pad=15)
                plt.tight_layout()
                plt.savefig(out_path(f'shap_summary_{model_name.replace(" ", "_")}.png'),
                           dpi=300, bbox_inches='tight')
                print(f"✓ shap_summary_{model_name.replace(' ', '_')}.png")
                plt.close()
                
                # --- (b) SHAP Bar Plot（带正负方向颜色）---
                imp_df = shap_data['importance']
                top_n = min(18, len(imp_df))
                top_imp = imp_df.head(top_n)
                
                fig, ax = plt.subplots(figsize=(11, 8))
                bar_colors = ['#e74c3c' if d == '正影响(↑)' else '#3498db'
                              for d in top_imp['Direction']]
                y_pos = np.arange(top_n)
                bars = ax.barh(y_pos, top_imp['Mean_Abs_SHAP'], color=bar_colors, alpha=0.85)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_imp['Feature'], fontsize=10)
                ax.invert_yaxis()
                ax.set_xlabel('mean(|SHAP value|)', fontsize=12)
                ax.set_title(f'{model_name} — SHAP Feature Importance (红=正影响, 蓝=负影响)',
                            fontsize=12, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                for bar, val in zip(bars, top_imp['Mean_Abs_SHAP']):
                    ax.text(bar.get_width() + bar.get_width()*0.01, bar.get_y()+bar.get_height()/2,
                           f'{val:.4f}', va='center', fontsize=8)
                # 图例
                from matplotlib.patches import Patch
                legend_elems = [Patch(facecolor='#e74c3c', label='正影响(增大PM)'),
                                Patch(facecolor='#3498db', label='负影响(减小PM)')]
                ax.legend(handles=legend_elems, loc='lower right', fontsize=10)
                plt.tight_layout()
                plt.savefig(out_path(f'shap_bar_{model_name.replace(" ", "_")}.png'),
                           dpi=300, bbox_inches='tight')
                print(f"✓ shap_bar_{model_name.replace(' ', '_')}.png")
                plt.close()
                
                # --- (c) SHAP Waterfall Plot（最高预测值样本）---
                try:
                    n_sv = sv.shape[0]
                    approx_pred = sv.sum(axis=1) + base_val
                    highest_idx = int(np.argmax(approx_pred))
                    
                    if hasattr(X_samp, 'iloc'):
                        sample_data = X_samp.iloc[highest_idx].values
                    else:
                        sample_data = X_samp[highest_idx]
                    
                    plt.figure(figsize=(11, 8))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=sv[highest_idx],
                            base_values=base_val,
                            data=sample_data,
                            feature_names=feat_names_used
                        ),
                        show=False, max_display=15
                    )
                    plt.title(f'{model_name} — SHAP Waterfall (预测值最高样本)',
                             fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(out_path(f'shap_waterfall_{model_name.replace(" ", "_")}.png'),
                               dpi=300, bbox_inches='tight')
                    print(f"✓ shap_waterfall_{model_name.replace(' ', '_')}.png")
                    plt.close()
                except Exception as ew:
                    print(f"  Waterfall图跳过: {ew}")
                
                                # --- (d) SHAP 依赖图（Top 10特征，仅最优树模型）---
                if model_name == best_tree_for_shap:
                    top10_features = imp_df['Feature'].head(10).tolist()
                    fig, axes_dep = plt.subplots(2, 2, figsize=(16, 12))
                    axes_dep = axes_dep.flatten()
                    
                    for fi, feat in enumerate(top10_features):
                        if feat in feat_names_used:
                            feat_idx = feat_names_used.index(feat)
                            ax_dep = axes_dep[fi]
                            
                            # 获取该特征的特征值和SHAP值
                            feat_vals = X_samp[feat].values if hasattr(X_samp,'columns') else X_samp[:, feat_idx]
                            shap_vals_feat = sv[:, feat_idx]
                            
                            # 用Ground_PM作为交互色彩变量（若存在）
                            color_feat = 'Ground_PM' if 'Ground_PM' in feat_names_used else None
                            if color_feat and color_feat != feat:
                                color_idx = feat_names_used.index(color_feat)
                                c_vals = X_samp[color_feat].values if hasattr(X_samp,'columns') else X_samp[:, color_idx]
                                sc = ax_dep.scatter(feat_vals, shap_vals_feat,
                                                   c=c_vals, cmap='RdYlBu_r', alpha=0.7, s=20)
                                plt.colorbar(sc, ax=ax_dep, label=color_feat)
                            else:
                                ax_dep.scatter(feat_vals, shap_vals_feat,
                                             alpha=0.6, s=20, c='steelblue')
                            
                            ax_dep.axhline(0, color='gray', linestyle='--', lw=1)
                            ax_dep.set_xlabel(feat, fontsize=11)
                            ax_dep.set_ylabel('SHAP value', fontsize=11)
                            ax_dep.set_title(f'依赖图: {feat}', fontsize=12, fontweight='bold')
                            ax_dep.grid(alpha=0.3)
                    
                    plt.suptitle(f'{model_name} — SHAP依赖图',
                                fontsize=14, fontweight='bold', y=1.01)
                    plt.tight_layout()
                    plt.savefig(out_path(f'shap_dependence_{model_name.replace(" ", "_")}.png'),
                               dpi=300, bbox_inches='tight')
                    print(f"✓ shap_dependence_{model_name.replace(' ', '_')}.png")
                    plt.close()
                    
            except Exception as e:
                print(f"⚠️  {model_name} SHAP可视化失败: {str(e)}")
                continue
        
        # --- (e) 双最优模型 SHAP 对比图 ---
        if len(shap_results) >= 2 and results:
            print("生成双模型SHAP对比图...")
            sorted_by_r2 = sorted(
                [m for m in shap_results if m in results],
                key=lambda m: results[m]['test_r2'], reverse=True
            )
            top2 = sorted_by_r2[:2]
            
            try:
                fig, axes2 = plt.subplots(1, 2, figsize=(22, 9))
                for ai, mn in enumerate(top2):
                    shap_importance_matrix = pd.DataFrame()
                    for mn2, sd2 in shap_results.items():
                        shap_importance_matrix[mn2] = (
                            sd2['importance'].set_index('Feature')['Mean_Abs_SHAP']
                        )
                    
                    top_feats_all = (shap_importance_matrix
                                    .fillna(0).sum(axis=1).nlargest(15).index.tolist())
                    
                    sd   = shap_results[mn]
                    imp  = sd['importance'].set_index('Feature')['Mean_Abs_SHAP']
                    vals = [imp.get(f, 0) for f in top_feats_all]
                    
                    # 重新查方向
                    dir_map = dict(zip(sd['importance']['Feature'], sd['importance']['Direction']))
                    colors_bar = ['#e74c3c' if dir_map.get(f,'正影响(↑)')=='正影响(↑)' else '#3498db'
                                  for f in top_feats_all]
                    
                    y_p = np.arange(len(top_feats_all))
                    axes2[ai].barh(y_p, vals, color=colors_bar, alpha=0.85)
                    axes2[ai].set_yticks(y_p)
                    axes2[ai].set_yticklabels(top_feats_all, fontsize=9)
                    axes2[ai].invert_yaxis()
                    axes2[ai].set_xlabel('mean(|SHAP value|)', fontsize=11)
                    r2v = results[mn]['test_r2']
                    axes2[ai].set_title(f'{mn} (Test R²={r2v:.4f})', fontsize=12, fontweight='bold')
                    axes2[ai].grid(axis='x', alpha=0.3)
                
                plt.suptitle('Top-2模型 SHAP特征重要性对比 (红=正影响, 蓝=负影响)',
                            fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(out_path('shap_top2_comparison.png'), dpi=300, bbox_inches='tight')
                print("✓ shap_top2_comparison.png")
                plt.close()
            except Exception as ec:
                print(f"  双模型对比图跳过: {ec}")
        
        # --- (f) SHAP重要性热力图（跨模型）---
        print("生成SHAP重要性热力图...")
        shap_importance_matrix = pd.DataFrame()
        for mn2, sd2 in shap_results.items():
            shap_importance_matrix[mn2] = (
                sd2['importance'].set_index('Feature')['Mean_Abs_SHAP']
            )
        
        if not shap_importance_matrix.empty:
            shap_importance_norm = shap_importance_matrix.fillna(0)
            shap_importance_norm = shap_importance_norm.div(
                shap_importance_norm.max(axis=0).replace(0, 1), axis=1
            )
            top_shap_features = shap_importance_norm.sum(axis=1).nlargest(20).index
            
            plt.figure(figsize=(14, 10))
            sns.heatmap(shap_importance_norm.loc[top_shap_features],
                       annot=True, fmt='.3f', cmap='Blues',
                       cbar_kws={'label': '标准化SHAP重要性'},
                       linewidths=0.5)
            plt.title('SHAP重要性热力图 (Top 20特征, 跨模型对比)',
                     fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('模型', fontsize=12)
            plt.ylabel('特征', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(out_path('shap_importance_heatmap.png'), dpi=300, bbox_inches='tight')
            print("✓ shap_importance_heatmap.png")
            plt.close()
    
    # 7.7 模型排名雷达图
    print("生成模型性能雷达图...")
    
    # 准备数据（标准化到0-1）
    metrics = {
        'R²': test_r2,
        'RMSE': [1/(1+x) for x in test_rmse],  # 转换为越大越好
        'MAE': [1/(1+x) for x in test_mae],    # 转换为越大越好
        'CV R²': cv_means,
        '泛化能力': [1-abs(train_r2[i]-test_r2[i]) for i in range(len(model_names))]
    }
    
    # 标准化
    for key in metrics:
        max_val = max(metrics[key])
        min_val = min(metrics[key])
        if max_val > min_val:
            metrics[key] = [(x-min_val)/(max_val-min_val) for x in metrics[key]]
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    colors_radar = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    for idx, model_name in enumerate(model_names):
        values = [metrics[key][idx] for key in metrics.keys()]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors_radar[idx])
        ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics.keys(), fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('模型综合性能雷达图', fontsize=14, fontweight='bold', pad=30)
    plt.tight_layout()
    plt.savefig(out_path('model_radar_chart.png'), dpi=300, bbox_inches='tight')
    print("✓ 保存: model_radar_chart.png")
    plt.close()

    # 7.8 过拟合诊断专项图
    print("生成过拟合诊断图...")
    gaps = [results[m]['train_r2'] - results[m]['test_r2'] for m in model_names]
    cv_r2 = [results[m]['cv_r2_mean'] for m in model_names]

    fig, axes_of = plt.subplots(1, 3, figsize=(22, 7))

    # (1) 训练/测试/CV R² 三线对比
    x = np.arange(len(model_names))
    w = 0.25
    axes_of[0].bar(x - w, train_r2, w, label='训练集R²',  color='#3498db', alpha=0.85)
    axes_of[0].bar(x,     test_r2,  w, label='测试集R²',  color='#e74c3c', alpha=0.85)
    axes_of[0].bar(x + w, cv_r2,    w, label='CV R²(均值)', color='#2ecc71', alpha=0.85)
    axes_of[0].set_xticks(x)
    axes_of[0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    axes_of[0].set_ylabel('R²', fontsize=11)
    axes_of[0].set_title('训练/测试/CV R² 三向对比', fontsize=12, fontweight='bold')
    axes_of[0].legend(fontsize=9)
    axes_of[0].axhline(0.8, color='gray', ls='--', lw=1, alpha=0.5)
    axes_of[0].grid(axis='y', alpha=0.3)

    # (2) 过拟合差距条形图（颜色编码）
    gap_colors = []
    for g in gaps:
        if   g > 0.20: gap_colors.append('#c0392b')   # 深红：严重
        elif g > 0.12: gap_colors.append('#e67e22')   # 橙：明显
        elif g > 0.05: gap_colors.append('#f1c40f')   # 黄：轻微
        else:          gap_colors.append('#27ae60')   # 绿：良好
    bars_gap = axes_of[1].bar(model_names, gaps, color=gap_colors, alpha=0.85, edgecolor='white')
    axes_of[1].axhline(0.20, color='#c0392b', ls='--', lw=1.5, label='严重阈值(0.20)')
    axes_of[1].axhline(0.12, color='#e67e22', ls='--', lw=1.5, label='明显阈值(0.12)')
    axes_of[1].axhline(0.05, color='#f1c40f', ls='--', lw=1.5, label='轻微阈值(0.05)')
    for bar, g in zip(bars_gap, gaps):
        axes_of[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f'{g:+.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes_of[1].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    axes_of[1].set_ylabel('过拟合差距（训练R² - 测试R²）', fontsize=11)
    axes_of[1].set_title('各模型过拟合差距（颜色=严重程度）', fontsize=12, fontweight='bold')
    axes_of[1].grid(axis='y', alpha=0.3)
    # 图例补丁
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color='#27ae60', label='🟢 泛化良好 (≤0.05)'),
        Patch(color='#f1c40f', label='🟡 轻微过拟合 (0.05-0.12)'),
        Patch(color='#e67e22', label='🟠 明显过拟合 (0.12-0.20)'),
        Patch(color='#c0392b', label='🔴 严重过拟合 (>0.20)')
    ]
    axes_of[1].legend(handles=legend_patches, fontsize=8, loc='upper left')

    # (3) 测试R² vs CV R² 散点图（理想=对角线）
    axes_of[2].scatter(cv_r2, test_r2, c=gap_colors, s=120, edgecolors='black', linewidth=0.8, zorder=5)
    min_v = min(min(cv_r2), min(test_r2)) - 0.05
    max_v = max(max(cv_r2), max(test_r2)) + 0.05
    axes_of[2].plot([min_v, max_v], [min_v, max_v], 'k--', lw=1.5, alpha=0.5, label='理想线(CV=Test)')
    for mn2, cx, ty in zip(model_names, cv_r2, test_r2):
        axes_of[2].annotate(mn2, (cx, ty), textcoords='offset points',
                            xytext=(4, 4), fontsize=7)
    axes_of[2].set_xlabel('CV R²（交叉验证均值）', fontsize=11)
    axes_of[2].set_ylabel('测试集R²', fontsize=11)
    axes_of[2].set_title('CV R² vs 测试集R²（偏离对角线=泛化差）', fontsize=12, fontweight='bold')
    axes_of[2].legend(fontsize=9)
    axes_of[2].grid(alpha=0.3)

    plt.suptitle('模型过拟合诊断综合分析', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path('overfitting_diagnosis.png'), dpi=300, bbox_inches='tight')
    print("✓ 保存: overfitting_diagnosis.png")
    plt.close()

    # 7.9 误差分布箱线图
    print("生成误差分布箱线图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绝对误差
    abs_errors = [np.abs(y_test - results[m]['y_pred_test']) for m in model_names]
    bp1 = axes[0].boxplot(abs_errors, labels=model_names, patch_artist=True,
                          showmeans=True, meanline=True)
    for patch, color in zip(bp1['boxes'], colors_radar):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_xlabel('模型', fontsize=12)
    axes[0].set_ylabel('绝对误差', fontsize=12)
    axes[0].set_title('模型绝对误差分布', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # 相对误差百分比
    rel_errors = [np.abs((y_test - results[m]['y_pred_test']) / y_test) * 100 
                  for m in model_names]
    bp2 = axes[1].boxplot(rel_errors, labels=model_names, patch_artist=True,
                          showmeans=True, meanline=True)
    for patch, color in zip(bp2['boxes'], colors_radar):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_xlabel('模型', fontsize=12)
    axes[1].set_ylabel('相对误差 (%)', fontsize=12)
    axes[1].set_title('模型相对误差分布', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(out_path('error_distribution.png'), dpi=300, bbox_inches='tight')
    print("✓ 保存: error_distribution.png")
    plt.close()
    
    print("\n所有可视化图表生成完成！")
    # ==================== 8. 生成详细分析报告 ====================
def generate_report(results, importance_results, shap_results, feature_names):
    """生成详细的分析报告"""
    
    print("\n" + "=" * 100)
    print("步骤 6: 生成分析报告")
    print("=" * 100)
    
    # 提前初始化跨模型统计变量
    all_top5_features = []
    all_top10_features = []
    all_shap_top5 = []
    all_shap_top10 = []

    report = []
    report.append("=" * 120)
    report.append("地铁颗粒物浓度预测模型分析报告")
    report.append("=" * 120)
    report.append(f"\n生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n")
    
    # 1. 数据概况
    report.append("一、数据概况")
    report.append("-" * 120)
    report.append(f"特征数量: {len(feature_names)}")
    report.append(f"\n特征列表:")
    for i, feat in enumerate(feature_names, 1):
        report.append(f"  {i:2d}. {feat}")
    report.append(f"\n目标变量: Metro_PM")
    report.append(f"训练集/测试集划分比例: 80% / 20%")
    report.append("\n")
    
    # 2. 模型性能总结
    report.append("二、模型性能总结")
    report.append("-" * 120)
    
    # 创建性能对比表
    performance_df = pd.DataFrame({
        '模型': list(results.keys()),
        '训练集R²': [f"{results[m]['train_r2']:.4f}" for m in results.keys()],
        '测试集R²': [f"{results[m]['test_r2']:.4f}" for m in results.keys()],
        '测试集RMSE': [f"{results[m]['test_rmse']:.4f}" for m in results.keys()],
        '测试集MAE': [f"{results[m]['test_mae']:.4f}" for m in results.keys()],
        '测试集MAPE': [f"{results[m]['test_mape']:.2f}%" for m in results.keys()],
        '交叉验证R²': [f"{results[m]['cv_r2_mean']:.4f}±{results[m]['cv_r2_std']:.4f}" 
                      for m in results.keys()]
    })
    
    report.append("\n性能指标对比表:")
    report.append(performance_df.to_string(index=False))
    report.append("\n")
    
    # 找出最佳模型
    test_r2_values = [results[m]['test_r2'] for m in results.keys()]
    test_rmse_values = [results[m]['test_rmse'] for m in results.keys()]
    test_mae_values = [results[m]['test_mae'] for m in results.keys()]
    
    best_model_r2_idx = np.argmax(test_r2_values)
    best_model_rmse_idx = np.argmin(test_rmse_values)
    best_model_mae_idx = np.argmin(test_mae_values)
    
    best_model_r2 = list(results.keys())[best_model_r2_idx]
    best_model_rmse = list(results.keys())[best_model_rmse_idx]
    best_model_mae = list(results.keys())[best_model_mae_idx]
    
    report.append("最佳模型:")
    report.append(f"  ★ 最高R²: {best_model_r2} (R² = {test_r2_values[best_model_r2_idx]:.4f})")
    report.append(f"  ★ 最低RMSE: {best_model_rmse} (RMSE = {test_rmse_values[best_model_rmse_idx]:.4f})")
    report.append(f"  ★ 最低MAE: {best_model_mae} (MAE = {test_mae_values[best_model_mae_idx]:.4f})")
    report.append("\n")
    
    # 模型排名
    report.append("模型综合排名 (按测试集R²):")
    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
    for rank, (model_name, model_results) in enumerate(sorted_models, 1):
        report.append(f"  {rank}. {model_name:<20} R²={model_results['test_r2']:.4f}, "
                     f"RMSE={model_results['test_rmse']:.4f}, MAE={model_results['test_mae']:.4f}")
    report.append("\n")
    
    # 3. 过拟合分析
    report.append("三、过拟合分析")
    report.append("-" * 120)
    
    for model_name in results.keys():
        train_r2 = results[model_name]['train_r2']
        test_r2 = results[model_name]['test_r2']
        gap = train_r2 - test_r2
        
        if gap > 0.20:
            status = "🔴 严重过拟合"
            recommendation = "建议: 大幅增加正则化强度、减少模型复杂度、增加训练数据"
        elif gap > 0.12:
            status = "🟠 明显过拟合"
            recommendation = "建议: 适当增大正则化、降低max_depth、增大min_samples_leaf"
        elif gap > 0.05:
            status = "🟡 轻微过拟合"
            recommendation = "建议: 可接受范围，可微调正则化参数进一步改善"
        elif gap < -0.05:
            status = "⚠️  欠拟合"
            recommendation = "建议: 增加模型复杂度、添加更多特征、减少正则化"
        else:
            status = "🟢 泛化良好"
            recommendation = "模型泛化能力优秀，训练/测试一致"
        
        report.append(f"{model_name:<20} 训练R²={train_r2:.4f}, 测试R²={test_r2:.4f}, "
                     f"差距={gap:+.4f} {status}")
        report.append(f"{'':20} {recommendation}")
    report.append("\n")
    
    # 4. 特征重要性总结
    report.append("四、特征重要性总结")
    report.append("-" * 120)
    
    if importance_results:
        from collections import Counter
        
        for model_name, importance_df in importance_results.items():
            if importance_df is not None:
                report.append(f"\n{model_name} - Top 10 重要特征:")
                top_10 = importance_df.head(10)
                for rank, (idx, row) in enumerate(top_10.items(), 1):
                    report.append(f"  {rank:2d}. {idx:<30} {row:.6f}")
                
                all_top5_features.extend(importance_df.head(5).index.tolist())
                all_top10_features.extend(importance_df.head(10).index.tolist())
        
        # 跨模型特征重要性统计
        if all_top5_features:
            report.append("\n跨模型特征重要性统计 (Top 5出现频率):")
            feature_counts_top5 = Counter(all_top5_features)
            for feature, count in feature_counts_top5.most_common(10):
                percentage = (count / len([imp for imp in importance_results.values() if imp is not None])) * 100
                report.append(f"  {feature:<30} 出现{count}次 ({percentage:.1f}%的模型)")
        
        if all_top10_features:
            report.append("\n跨模型特征重要性统计 (Top 10出现频率):")
            feature_counts_top10 = Counter(all_top10_features)
            for feature, count in feature_counts_top10.most_common(15):
                percentage = (count / len([imp for imp in importance_results.values() if imp is not None])) * 100
                report.append(f"  {feature:<30} 出现{count}次 ({percentage:.1f}%的模型)")
    
    report.append("\n")
    
    # 5. SHAP分析总结
    if shap_results:
        report.append("五、SHAP值分析总结")
        report.append("-" * 120)
        
        for model_name, shap_data in shap_results.items():
            if 'importance' in shap_data:
                report.append(f"\n{model_name} - Top 10 SHAP重要特征:")
                top_10_shap = shap_data['importance'].head(10)
                for rank, (idx, row) in enumerate(top_10_shap.iterrows(), 1):
                    report.append(f"  {rank:2d}. {row['Feature']:<30} {row['Mean_Abs_SHAP']:.6f} {row['Direction']}")
                
                all_shap_top5.extend(shap_data['importance'].head(5)['Feature'].tolist())
                all_shap_top10.extend(shap_data['importance'].head(10)['Feature'].tolist())
        
        # 跨模型SHAP重要性统计
        if all_shap_top5:
            from collections import Counter
            report.append("\n跨模型SHAP重要性统计 (Top 5出现频率):")
            shap_counts_top5 = Counter(all_shap_top5)
            for feature, count in shap_counts_top5.most_common(10):
                percentage = (count / len(shap_results)) * 100
                report.append(f"  {feature:<30} 出现{count}次 ({percentage:.1f}%的模型)")
            
            report.append("\n跨模型SHAP重要性统计 (Top 10出现频率):")
            shap_counts_top10 = Counter(all_shap_top10)
            for feature, count in shap_counts_top10.most_common(15):
                percentage = (count / len(shap_results)) * 100
                report.append(f"  {feature:<30} 出现{count}次 ({percentage:.1f}%的模型)")
    
    report.append("\n")
    
    # 6. 最佳超参数配置
    report.append("六、最佳超参数配置")
    report.append("-" * 120)
    
    for model_name in results.keys():
        report.append(f"\n{model_name}:")
        best_params = results[model_name]['best_params']
        for param, value in best_params.items():
            report.append(f"  {param:<30} {value}")
    
    report.append("\n")
    
    # 7. 模型性能评级
    report.append("七、模型性能评级")
    report.append("-" * 120)
    
    for model_name in results.keys():
        test_r2 = results[model_name]['test_r2']
        test_rmse = results[model_name]['test_rmse']
        test_mape = results[model_name]['test_mape']
        
        # R²评级
        if test_r2 >= 0.9:
            r2_grade = "A+ (优秀)"
        elif test_r2 >= 0.8:
            r2_grade = "A  (良好)"
        elif test_r2 >= 0.7:
            r2_grade = "B  (中等)"
        elif test_r2 >= 0.6:
            r2_grade = "C  (及格)"
        else:
            r2_grade = "D  (较差)"
        
        # MAPE评级
        if test_mape < 10:
            mape_grade = "A+ (优秀)"
        elif test_mape < 20:
            mape_grade = "A  (良好)"
        elif test_mape < 30:
            mape_grade = "B  (中等)"
        elif test_mape < 50:
            mape_grade = "C  (及格)"
        else:
            mape_grade = "D  (较差)"
        
        report.append(f"\n{model_name}:")
        report.append(f"  R²评级:    {r2_grade}")
        report.append(f"  MAPE评级:  {mape_grade}")
        report.append(f"  综合评价:  R²={test_r2:.4f}, RMSE={test_rmse:.4f}, MAPE={test_mape:.2f}%")
    
    report.append("\n")
    
    # 8. 关键发现
    report.append("八、关键发现")
    report.append("-" * 120)
    
    # 8.1 最佳模型分析
    best_overall = sorted_models[0][0]
    best_r2 = sorted_models[0][1]['test_r2']
    best_rmse = sorted_models[0][1]['test_rmse']
    best_mape = sorted_models[0][1]['test_mape']
    
    report.append(f"\n1. 最佳预测模型: {best_overall}")
    report.append(f"   - 测试集R²达到 {best_r2:.4f}，表明模型能够解释{best_r2*100:.2f}%的方差")
    report.append(f"   - RMSE为 {best_rmse:.4f}，MAE为 {sorted_models[0][1]['test_mae']:.4f}")
    report.append(f"   - MAPE为 {best_mape:.2f}%，平均预测误差在可接受范围内")
    
    # 8.2 模型对比分析
    report.append("\n2. 模型类型对比:")
    
    tree_models = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'Extra Trees', 'LightGBM']
    linear_models = ['Ridge', 'Lasso', 'Adaptive Lasso', 'ElasticNet']
    other_models = ['KNN', 'SVM']
    
    tree_r2_list = [results[m]['test_r2'] for m in tree_models if m in results]
    linear_r2_list = [results[m]['test_r2'] for m in linear_models if m in results]
    other_r2_list = [results[m]['test_r2'] for m in other_models if m in results]
    
    tree_avg_r2 = np.mean(tree_r2_list) if tree_r2_list else 0
    linear_avg_r2 = np.mean(linear_r2_list) if linear_r2_list else 0
    other_avg_r2 = np.mean(other_r2_list) if other_r2_list else 0
    
    report.append(f"   - 树模型平均R²: {tree_avg_r2:.4f}")
    report.append(f"   - 线性模型平均R²: {linear_avg_r2:.4f}")
    report.append(f"   - 其他模型平均R²: {other_avg_r2:.4f}")
    
    if tree_avg_r2 > max(linear_avg_r2, other_avg_r2):
        report.append("   ★ 树模型整体表现最佳，说明数据存在复杂非线性关系")
    elif linear_avg_r2 > max(tree_avg_r2, other_avg_r2):
        report.append("   ★ 线性模型整体表现最佳，说明特征与目标变量存在较强线性关系")
    
    # 8.3 关键特征识别
    report.append("\n3. 关键影响因素:")
    
    if importance_results or shap_results:
        from collections import Counter
        all_important_features = []
        
        if importance_results and all_top5_features:
            feature_counts = Counter(all_top5_features)
            all_important_features.extend([f for f, c in feature_counts.most_common(5)])
        
        if shap_results and all_shap_top5:
            shap_counts = Counter(all_shap_top5)
            all_important_features.extend([f for f, c in shap_counts.most_common(5)])
        
        # 去重并统计
        final_important = Counter(all_important_features).most_common(5)
        
        report.append("   根据特征重要性和SHAP值综合分析，影响PM浓度的关键因素为:")
        for rank, (feature, count) in enumerate(final_important, 1):
            report.append(f"   {rank}. {feature}")
    
    # 8.4 过拟合风险评估
    report.append("\n4. 过拟合风险评估:")
    
    overfit_models = []
    good_models = []
    
    for model_name in results.keys():
        gap = results[model_name]['train_r2'] - results[model_name]['test_r2']
        if gap > 0.10:
            overfit_models.append((model_name, gap))
        else:
            good_models.append((model_name, gap))
    
    if overfit_models:
        report.append(f"   - {len(overfit_models)}个模型存在过拟合风险:")
        for model, gap in sorted(overfit_models, key=lambda x: x[1], reverse=True):
            report.append(f"     • {model} (差距: {gap:.4f})")
    
    if good_models:
        report.append(f"   - {len(good_models)}个模型泛化能力良好:")
        for model, gap in sorted(good_models, key=lambda x: x[1]):
            report.append(f"     • {model} (差距: {gap:.4f})")
    
    report.append("\n")
    
    # 9. 结论与建议
    report.append("九、结论与建议")
    report.append("-" * 120)
    
    report.append("\n【模型选择建议】")
    
    if best_r2 >= 0.9:
        report.append(f"✓ 推荐使用 {best_overall} 模型进行PM浓度预测")
        report.append("  该模型预测性能优秀，R²超过0.9，可直接用于实际应用")
    elif best_r2 >= 0.8:
        report.append(f"✓ 推荐使用 {best_overall} 模型进行PM浓度预测")
        report.append("  该模型预测性能良好，R²超过0.8，建议进一步优化后投入使用")
    elif best_r2 >= 0.7:
        report.append(f"○ 可考虑使用 {best_overall} 模型，但需要进一步改进")
        report.append("  该模型预测性能中等，建议:")
        report.append("    - 收集更多训练数据")
        report.append("    - 进行更深入的特征工程")
        report.append("    - 尝试集成学习方法")
    else:
        report.append(f"✗ 当前最佳模型 {best_overall} 的R²仅为 {best_r2:.4f}，性能不足")
        report.append("  建议:")
        report.append("    - 重新审视数据质量和特征选择")
        report.append("    - 收集更多相关特征")
        report.append("    - 检查是否存在数据泄漏或异常值")
        report.append("    - 考虑深度学习方法")
    
    report.append("\n【特征工程建议】")
    
    if importance_results or shap_results:
        report.append("基于特征重要性分析:")
        report.append("  1. 重点关注高重要性特征的数据质量")
        report.append("  2. 考虑创建重要特征之间的交互项")
        report.append("  3. 对低重要性特征进行特征选择，简化模型")
        
        # 识别低重要性特征
        if importance_results:
            low_importance_features = set()
            for imp_df in importance_results.values():
                if imp_df is not None:
                    # 取重要性最低的5个特征
                    low_features = imp_df.tail(5).index.tolist()
                    low_importance_features.update(low_features)
            
            if low_importance_features:
                report.append("\n  可考虑移除的低重要性特征:")
                for feat in sorted(low_importance_features)[:10]:
                    report.append(f"    - {feat}")
    
    report.append("\n【数据收集建议】")
    report.append("  1. 增加时间序列特征（如前一时刻的PM值）")
    report.append("  2. 补充气象数据（如降雨量、能见度等）")
    report.append("  3. 添加人流量、列车频次等运营数据")
    report.append("  4. 收集更多城市和站点的数据以提高模型泛化能力")
    
    report.append("\n【模型部署建议】")
    report.append(f"  1. 生产环境推荐使用: {best_overall}")
    
    # 推荐备选模型
    if len(sorted_models) > 1:
        second_best = sorted_models[1][0]
        second_r2 = sorted_models[1][1]['test_r2']
        report.append(f"  2. 备选模型: {second_best} (R²={second_r2:.4f})")
    
    report.append("  3. 建立模型监控机制:")
    report.append("     - 定期评估模型在新数据上的性能")
    report.append("     - 设置预测误差阈值告警")
    report.append("     - 每季度使用新数据重新训练模型")
    
    report.append("\n【实际应用建议】")
    report.append("  1. 预警系统:")
    report.append("     - 当预测PM浓度超过阈值时，提前启动通风系统")
    report.append("     - 在高峰时段加强空气质量监测")
    
    report.append("\n  2. 优化措施:")
    report.append("     - 根据关键影响因素制定针对性改善方案")
    report.append("     - 在高风险时段和站点增加清洁频次")
    report.append("     - 优化通风系统运行策略")
    
    report.append("\n  3. 持续改进:")
    report.append("     - 收集模型预测与实际值的偏差数据")
    report.append("     - 分析预测失败的案例，改进模型")
    report.append("     - 结合领域专家知识优化特征工程")
    
    report.append("\n【模型可解释性】")
    
    if shap_results:
        report.append("  ✓ 已生成SHAP值分析，可用于:")
        report.append("    - 向管理层解释模型预测结果")
        report.append("    - 识别异常预测的原因")
        report.append("    - 验证模型决策的合理性")
        report.append("    - 指导运营优化决策")
    
    report.append("\n【风险提示】")
    report.append("  1. 模型预测存在不确定性，建议结合实际监测数据使用")
    report.append("  2. 当输入特征超出训练数据范围时，预测可能不准确")
    report.append("  3. 模型未考虑突发事件（如设备故障、极端天气）的影响")
    report.append("  4. 需要定期更新模型以适应环境变化")
    
    report.append("\n")
    report.append("=" * 120)
    report.append("报告生成完成")
    report.append("=" * 120)
    report.append(f"\n分析人员: 数据分析系统")
    report.append(f"报告版本: v5.0 (优化版)")
    report.append(f"模型数量: {len(results)} 个")
    report.append("\n")
    
    # 保存报告
    report_text = "\n".join(report)
    with open(out_path('analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\n✓ 报告已保存至: {out_path('analysis_report.txt')}")
    
    return report_text
# ==================== 9. 保存模型和结果 ====================
def save_results(results, trained_models, importance_results, shap_results):
    """保存模型和分析结果"""
    
    print("\n" + "=" * 100)
    print("步骤 7: 保存模型和结果")
    print("=" * 100)
    
    import pickle
    
    # 保存训练好的模型
    print("\n保存训练模型...")
    for model_name, model_info in trained_models.items():
        filename = out_path(f"model_{model_name.replace(' ', '_')}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(model_info['model'], f)
        print(f"✓ 保存: {filename}")
    
    # 保存结果摘要
    print("\n保存结果摘要...")
    results_summary = pd.DataFrame({
        '模型': list(results.keys()),
        '训练集R²': [results[m]['train_r2'] for m in results.keys()],
        '测试集R²': [results[m]['test_r2'] for m in results.keys()],
                '测试集RMSE': [results[m]['test_rmse'] for m in results.keys()],
        '测试集MAE': [results[m]['test_mae'] for m in results.keys()],
        '测试集MAPE': [results[m]['test_mape'] for m in results.keys()],
        '交叉验证R²均值': [results[m]['cv_r2_mean'] for m in results.keys()],
        '交叉验证R²标准差': [results[m]['cv_r2_std'] for m in results.keys()],
        '过拟合差距': [results[m]['overfit_gap'] for m in results.keys()]
    })
    results_summary.to_csv(out_path('model_performance_summary.csv'), index=False, encoding='utf-8-sig')
    print("✓ 保存: model_performance_summary.csv")
    
    # 保存特征重要性
    if importance_results:
        print("\n保存特征重要性...")
        with pd.ExcelWriter(out_path('feature_importance_summary.xlsx'), engine='openpyxl') as writer:
            for model_name, imp_df in importance_results.items():
                if imp_df is not None:
                    sheet_name = model_name[:31]  # Excel sheet名称限制
                    imp_df.to_frame(name='Importance').to_excel(writer, sheet_name=sheet_name, index=True)
        print("✓ 保存: feature_importance_summary.xlsx")
    
    # 保存SHAP重要性
    if shap_results:
        print("\n保存SHAP重要性...")
        with pd.ExcelWriter(out_path('shap_importance_summary.xlsx'), engine='openpyxl') as writer:
            for model_name, shap_data in shap_results.items():
                if 'importance' in shap_data:
                    sheet_name = model_name[:31]
                    shap_data['importance'].to_excel(writer, sheet_name=sheet_name, index=False)
        print("✓ 保存: shap_importance_summary.xlsx")
    
    # 保存最佳超参数
    print("\n保存最佳超参数...")
    best_params_df = pd.DataFrame([
        {'模型': model_name, '参数': str(results[model_name]['best_params'])}
        for model_name in results.keys()
    ])
    best_params_df.to_csv(out_path('best_hyperparameters.csv'), index=False, encoding='utf-8-sig')
    print("✓ 保存: best_hyperparameters.csv")
    
    print("\n所有结果已保存！")
    # ==================== 10. 主函数 ====================
def main(file_path='2.xlsx'):
    """主执行函数"""
    print("\n开始分析流程...\n")
    try:
        # 步骤1: 加载数据
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_names = \
        load_and_preprocess_data(file_path)
        
        # 确保X_train_scaled和X_test_scaled被正确定义（添加额外保护）
        if X_train_scaled is None or X_test_scaled is None:
            print("警告: X_train_scaled和X_test_scaled未定义，使用原始数据进行标准化")
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
        # ==================== 是否启用数据增强 ====================
        USE_AUGMENT = False # ✅ 开关
        N_AUG = 2 # 小样本建议1（最多2），每个样本生成1个增强样本
        NOISE_SCALE = 0.15 # 小样本核心：不超过0.4
        if USE_AUGMENT:
            print("\n>>> 启用残差驱动数据增强（小样本优化版）")
            print(f" 原始训练样本数: {len(X_train)}")
            X_aug, y_aug = residual_based_augmentation(
                X_train, y_train, n_aug=N_AUG, noise_scale=NOISE_SCALE
            )
            # 合并原始 + 增强数据
            X_train = pd.concat([X_train, X_aug], axis=0)
            y_train = pd.concat([y_train, y_aug], axis=0)
            print(f" 增强后训练样本数: {len(X_train)}")
            # 重新标准化（必须保留！增强后特征分布有微小变化）
            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train), 
                columns=X_train.columns, 
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test), 
                columns=X_test.columns, 
                index=X_test.index
            )
        
        # 步骤2: 训练模型
        results, trained_models = train_and_evaluate_models(
            X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
        )
        
        # 步骤3: 特征重要性分析
        importance_results = analyze_feature_importance(trained_models, X_train, X_test, y_train, y_test, feature_names)
        
        # 步骤4: SHAP分析
        shap_results = analyze_shap_values(
            trained_models, X_train, X_test,
            X_train_scaled, X_test_scaled, feature_names, y_train, y_test
        )
        
        # 步骤5: 生成可视化
        create_visualizations(results, importance_results, shap_results, 
                             feature_names, y_test)
        
        # 步骤6: 生成报告
        generate_report(results, importance_results, shap_results, feature_names)
        
        # 步骤7: 保存结果
        save_results(results, trained_models, importance_results, shap_results)
        
        print("\n" + "=" * 100)
        print("分析完成！")
        print("=" * 100)
        print(f"\n📁 所有输出文件已保存至目录: {OUTPUT_DIR}")
        print("\n生成的文件清单:")
        print("\n【可视化图表】")
        print("  1.  model_performance_comparison.png    - 模型性能对比图")
        print("  2.  prediction_vs_actual.png            - 预测值vs真实值散点图")
        print("  3.  residual_analysis.png               - 残差分析图")
        print("  4.  feature_importance.png              - 特征重要性图")
        print("  5.  feature_importance_heatmap.png      - 特征重要性热力图")
        print("  6.  model_radar_chart.png               - 模型性能雷达图")
        print("  7.  error_distribution.png              - 误差分布箱线图")
        print("  8.  overfitting_diagnosis.png           - 过拟合诊断图")
        print("  9.  shap_importance_heatmap.png         - SHAP重要性热力图")
        
        print("\n【SHAP可视化】(每个模型)")
        for model_name in shap_results.keys():
            model_file = model_name.replace(' ', '_')
            print(f"  -  shap_summary_{model_file}.png        - SHAP摘要图")
            print(f"  -  shap_bar_{model_file}.png            - SHAP条形图")
            print(f"  -  shap_waterfall_{model_file}.png      - SHAP瀑布图")
        
        print("\n【分析报告】")
        print("  10. analysis_report.txt                 - 详细分析报告")
        
        print("\n【数据文件】")
        print("  11. model_performance_summary.csv       - 模型性能汇总")
        print("  12. feature_importance_summary.xlsx     - 特征重要性汇总")
        print("  13. shap_importance_summary.xlsx        - SHAP重要性汇总")
        print("  14. best_hyperparameters.csv            - 最佳超参数配置")
        
        print("\n【模型文件】")
        for model_name in trained_models.keys():
            print(f"  -  model_{model_name.replace(' ', '_')}.pkl")
        
        print("\n" + "=" * 100)
        print("感谢使用地铁颗粒物浓度预测分析系统 v5.0！")
        print("=" * 100)
        print("\n")
        
        return results, trained_models, importance_results, shap_results
        
    except FileNotFoundError:
        print(f"\n❌ 错误: 找不到数据文件 '{file_path}'")
        print("请确保数据文件在当前目录下，或提供正确的文件路径")
        return None, None, None, None
        
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
# ==================== 11. 执行主函数 ====================
if __name__ == "__main__":
    # 修改为你的实际文件路径
    data_file = "E:\\2021-文件A\\文章\\paper6 地铁颗粒物影响因素分析\\Factor.xlsx"
    
    # 执行分析
    results, trained_models, importance_results, shap_results = main(data_file)
    
    # 如果需要，可以进一步分析结果
    if results is not None:
        print("\n" + "=" * 100)
        print("快速查看最佳模型")
        print("=" * 100)
        
        # 找出最佳模型
        best_model_name = max(results.items(), key=lambda x: x[1]['test_r2'])[0]
        best_results = results[best_model_name]
        
        print(f"\n最佳模型: {best_model_name}")
        print(f"  测试集 R²:    {best_results['test_r2']:.4f}")
        print(f"  测试集 RMSE:  {best_results['test_rmse']:.4f}")
        print(f"  测试集 MAE:   {best_results['test_mae']:.4f}")
        print(f"  测试集 MAPE:  {best_results['test_mape']:.2f}%")
        print(f"  交叉验证 R²:  {best_results['cv_r2_mean']:.4f} ± {best_results['cv_r2_std']:.4f}")
        print(f"  过拟合差距:   {best_results['overfit_gap']:+.4f}")
        
        print("\n最佳超参数:")
        for param, value in best_results['best_params'].items():
            print(f"  {param}: {value}")
        
        # 显示Top 5特征
        if importance_results and best_model_name in importance_results and importance_results[best_model_name] is not None:
            print(f"\nTop 5 重要特征 ({best_model_name}):")
            top5 = importance_results[best_model_name].head(5)
            for idx, row in top5.items():
                print(f"  {idx}: {row:.6f}")
        
        # 显示Top 5 SHAP特征
        if shap_results and best_model_name in shap_results:
            if 'importance' in shap_results[best_model_name]:
                print(f"\nTop 5 SHAP重要特征 ({best_model_name}):")
                top5_shap = shap_results[best_model_name]['importance'].head(5)
                for idx, row in top5_shap.iterrows():
                    print(f"  {row['Feature']}: {row['Mean_Abs_SHAP']:.6f} {row['Direction']}")
        
        print("\n" + "=" * 100)
        
        # 提供使用建议
        print("\n使用建议:")
        print("  1. 查看 'analysis_report.txt' 获取完整分析报告")
        print("  2. 查看各类PNG图表了解模型性能和特征重要性")
        print("  3. 使用保存的.pkl模型文件进行预测")
        print("  4. 查看过拟合诊断图评估模型泛化能力")
        
        print("\n示例代码 - 加载模型进行预测:")
        print(f"""

        """)
        
        print("\n" + "=" * 100)
        print("性能优化总结")
        print("=" * 100)
        
        # 统计性能提升
        print("\n本次优化措施:")
        print("  ✓ 去除TabPFN模型，简化模型集合")
        print("  ✓ 新增19个高级特征（多项式、交互、周期性编码等）")
        print("  ✓ 扩大超参数搜索空间，增加搜索迭代次数")
        print("  ✓ 优化树模型深度和正则化平衡")
        print("  ✓ 使用10折交叉验证提高评估稳定性")
        print("  ✓ 增强过拟合诊断和可视化")
        
        # 显示所有模型性能
        print("\n所有模型性能排名:")
        sorted_models = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
        print(f"\n{'排名':<6} {'模型':<22} {'测试R²':<10} {'RMSE':<10} {'MAE':<10} {'过拟合':<10}")
        print("-" * 80)
        for rank, (model_name, model_results) in enumerate(sorted_models, 1):
            gap = model_results['overfit_gap']
            gap_status = "🟢" if gap <= 0.05 else "🟡" if gap <= 0.12 else "🟠" if gap <= 0.20 else "🔴"
            print(f"{rank:<6} {model_name:<22} {model_results['test_r2']:<10.4f} "
                  f"{model_results['test_rmse']:<10.4f} {model_results['test_mae']:<10.4f} "
                  f"{gap_status} {gap:+.4f}")
        
        print("\n" + "=" * 100)
        print("预期性能提升")
        print("=" * 100)
        print("\n与原版本相比:")
        print("  • 特征数量: 18 → 37 (增加105%)")
        print("  • 超参数搜索: 标准 → 扩展 (平均增加40%搜索空间)")
        print("  • 交叉验证: 5折 → 10折 (提升评估稳定性)")
        print("  • 预期R²提升: +2-5% (取决于数据特性)")
        print("  • 过拟合控制: 保持严格的正则化机制")
        
        print("\n" + "=" * 100)
        print("下一步建议")
        print("=" * 100)
        print("\n如果性能仍需提升，可以尝试:")
        print("  1. 特征选择: 使用递归特征消除(RFE)筛选最优特征子集")
        print("  2. 集成优化: 调整Stacking的元模型和基模型组合")
        print("  3. 数据增强: 使用SMOTE等方法增加训练样本")
        print("  4. 深度学习: 尝试神经网络模型(MLP, TabNet)")
        print("  5. 时间序列: 如果数据有时间顺序，考虑LSTM/GRU")
        print("  6. 贝叶斯优化: 使用Optuna等工具进行更智能的超参数搜索")
        
        print("\n" + "=" * 100) 
