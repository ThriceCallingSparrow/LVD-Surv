"""lifetime 模块：提供项目内部的明确、可复用实现。"""
from scipy.stats import expon, weibull_min, norm, lognorm, gamma
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import joblib
import os

from lvd_surv.data.loader import CMAPSSLoader
from lvd_surv.runtime.cancellation import TaskCancelledError, check_cancelled

# 定义单一目标Distribution
DISTRIBUTIONS = {
    "expon": "指数Distribution",
    "weibull_min": "威布尔Distribution",
    "norm": "正态Distribution",
    "lognorm": "对数正态Distribution",
    "gamma": "伽马Distribution"
}

'''======== 1.5IQR 准则检查异常值 ========'''
def iqr_outlier_detection(data, verbose=True, return_cleaned=True):
    '''1.5IQR 准则检查异常值并画出箱线图'''
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    # 1.5IQR 阈值
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # 筛选正常寿命数据
    normal_life = data[(data >= lower) & (data <= upper)]
    outlier = data[(data < lower) | (data > upper)]
    
    if verbose:
        print(f"Q1={Q1}, Q3={Q3}, IQR={IQR}")
        print(f"Lower Bound={lower}, Upper Bound={upper}")
        print(f"Outlier Lifespan: {outlier}")
        print(f"Cleaned Valid Failure Lifespan: {normal_life}")
        
        # 绘制箱线图
        plt.boxplot(data)
        plt.title("Product Failure Lifespan Boxplot (1.5IQR Outlier Detection)")
        plt.savefig("analysis_result/boxplot.png", dpi=300)
        plt.close()
        
    if return_cleaned:
        return normal_life
    else:
        return data

'''======== 描述性统计分析 ========'''
def compute_basic_statistics(data, print_stats=True):
    '''计算统计量'''
    # 位置统计量
    min_val = np.min(data)       # Min
    max_val = np.max(data)       # Max
    median_val = np.median(data) # Median
    mean_val = np.mean(data)     # Mean
    
    # 离散统计量
    std_val = np.std(data, ddof=1)  # Sample Std
    cv_val = std_val / mean_val           # CV
    q1 = np.percentile(data, 25)   # 下四分位数 Q1
    q3 = np.percentile(data, 75)   # 上四分位数 Q3
    iqr_val = q3 - q1                     # IQR
    
    # 形状统计量
    skewness_val = stats.skew(data)                # 偏度
    kurtosis_val = stats.kurtosis(data, fisher=True) # 费舍尔峰度
    
    # 打印结果
    if print_stats:
        print("===== Location Statistics =====")
        print(f"Min: {min_val:.2f}")
        print(f"Max: {max_val:.2f}")
        print(f"Median: {median_val:.2f}")
        print(f"Mean: {mean_val:.2f}")
        
        print("\n===== Dispersion Statistics =====")
        print(f"Sample Std: {std_val:.4f}")
        print(f"CV: {cv_val:.4f}")
        print(f"IQR: {iqr_val:.4f}")
        
        print("\n===== Shape Statistics =====")
        print(f"Skewness: {skewness_val:.4f}")
        print(f"Kurtosis: {kurtosis_val:.4f}")
        
'''======== 经验Distribution函数计算 ========'''
def compute_empirical_failure_rates(data):
    '''计算经验失效率'''
    # 中位秩计算经验累积失效概率
    n =len(data)
    data = np.sort(data)
    i_arr = np.arange(1, n+1)  # 序号 i = 1,2...n
    F_hat = (i_arr - 0.3) / (n + 0.4)
    
    # 经验可靠度
    R_hat = 1 - F_hat
    
    # 差分求经验故障概率密度 f(t)
    f_hat = np.zeros_like(F_hat)
    dt = np.diff(data)
    dF = np.diff(F_hat)
    
    # 安全差分
    for i in range(1, len(f_hat)):
        current_dt = dt[i-1]
        current_dF = dF[i-1]
        
        if current_dt == 0:
            # 找最近的非零时间差做平均
            valid_dt = dt[dt != 0]
            if len(valid_dt) > 0:
                avg_dt = np.mean(valid_dt)
                f_hat[i] = current_dF / avg_dt
            else:
                # 不存在则使用上一时刻密度
                f_hat[i] = f_hat[i-1]
        else:
            # 正常差分计算
            f_hat[i] = current_dF / current_dt
            
    # 经验失效率 h(t)
    h_hat = f_hat / R_hat
    
    return h_hat, F_hat, R_hat, i_arr

def calculate_empirical_reliability_median_rank(data, print_table=True, plot_figure=True):
    '''计算经验可靠度和经验失效率'''
    data = np.sort(data)    # 从小到大排序
    h_hat, F_hat, R_hat, i_arr = compute_empirical_failure_rates(data)
    
    # 输出表格数据
    if print_table:
        print("\n" + "="*15 + " Lifetime Distribution Statistics " + "="*15)
        print(f"{'Index i':<6}{'Time t':<8}{'F(t)':<10}{'R(t)':<12}{'h(t)':<10}")
        for idx in range(len(data)):
            print(f"{i_arr[idx]:<6}{data[idx]:<8}{F_hat[idx]:<10.4f}{R_hat[idx]:<12.4f}{h_hat[idx]:<10.6f}")
    # 绘图
    if plot_figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # 可靠度曲线
        ax1.plot(data, R_hat, "o-", color="#2E86AB", linewidth=2, label=r"Empirical Reliability $\hat{R}(t)$")
        ax1.set_xlabel("Failure Time t")
        ax1.set_ylabel("Reliability R(t)")
        ax1.set_title("Empirical Reliability Curve (Median Rank Method)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        # 失效率曲线
        ax2.plot(data, h_hat, "s-", color="#A23B72", linewidth=2, label=r"Empirical Failure Rate $\hat{h}(t)$")
        ax2.set_xlabel("Failure Time t")
        ax2.set_ylabel("Failure Rate h(t)")
        ax2.set_title("Empirical Failure Rate Curve (Median Rank Method)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("analysis_result/empirical_reliability_failure_rate_curves.png", dpi=300, bbox_inches="tight")
        plt.close()
        
    return h_hat
    
'''======== 标准化拟合与检验流程 ========'''
def analyze_distribution(data, dist_en_name):
    """单一Distribution分析"""
    n = len(data)
    
    # 区分检验方法
    if dist_en_name in ["lognorm", "gamma"]:
        # 使用蒙特卡洛 AD 检验
        if dist_en_name == "lognorm":
            dist_class = stats.lognorm
        elif dist_en_name == "gamma":
            dist_class = stats.gamma
        
        # 蒙特卡洛 AD 检验
        res = stats.goodness_of_fit(        # H0: 样本服从指定理论Distribution
            dist_class,
            data,
            statistic="ad",
            known_params={"loc": 0},
            rng = np.random.default_rng(seed=42)
        )
        
        ad_stat = res.statistic
        pvalue = res.pvalue
        params = res.fit_result.params
        dist = dist_class(*params)
    # 其他Distribution正常使用 Anderson-Darling 检验
    else:
        # MLE拟合
        if dist_en_name == "weibull_min":
            params = stats.weibull_min.fit(data, floc=0)
        else:
            params = getattr(stats, dist_en_name).fit(data)
            
        dist = getattr(stats, dist_en_name)(*params)
        
        ad_res = stats.anderson(data, dist_en_name, method='interpolate')  # H0: 样本服从指定理论Distribution
        ad_stat = ad_res.statistic
        pvalue = ad_res.pvalue
        
    # 对数似然 / AIC / BIC
    k = len(params)
    log_likelihood = dist.logpdf(data).sum()
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    
    return {
        "zh_name": DISTRIBUTIONS[dist_en_name],
        "dist_en": dist_en_name,
        "params": np.round(params, 4),
        "ad_stat": round(ad_stat, 4),
        "pvalue": round(pvalue, 4),
        "aic": round(aic, 4),
        "bic": round(bic, 4),
        "dist_obj": dist
    }
    
# 绘制 P-P 图 & Q-Q 图
def plot_pp_qq(data, dist_results, file_prefix: str):
    """绘制所有候选Distribution的 P-P 图和 Q-Q 图"""
    n_dist = len(dist_results)
    fig, axes = plt.subplots(n_dist, 2, figsize=(10, 3 * n_dist))
    fig.suptitle("P-P Plot (Left)  &  Q-Q Plot (Right)", fontsize=16, y=0.98)
    
    for i, res in enumerate(dist_results):
        data_sorted = np.sort(data)
        dist = res["dist_obj"]
        
        # Q-Q 图: 理论分位数 VS 样本分位数
        quantiles = np.linspace(0.01, 0.99, len(data_sorted))
        theo_quantiles = dist.ppf(quantiles)
        axes[i,1].scatter(theo_quantiles, data_sorted, s=10, alpha=0.6, color='#2E86AB')
        axes[i,1].plot(theo_quantiles, theo_quantiles, 'r--', lw=2, label='y=x')
        axes[i,1].set_title(f"{res['dist_en']} Q-Q Plot", fontsize=12)
        axes[i,1].set_xlabel("Theoretical Quantiles")
        axes[i,1].set_ylabel("Sample Quantiles")
        axes[i,1].grid(alpha=0.3)
        axes[i,1].legend()
        
        # P-P 图: 理论累积概率 VS 经验累积概率
        emp_cdf = (np.arange(1, len(data_sorted)+1) - 0.3) / (len(data_sorted) + 0.4)
        theo_cdf = dist.cdf(data_sorted)
        axes[i,0].scatter(theo_cdf, emp_cdf, s=10, alpha=0.6, color='#A23B72')
        axes[i,0].plot([0,1], [0,1], 'r--', lw=2, label='y=x')
        axes[i,0].set_title(f"{res['dist_en']} P-P Plot", fontsize=12)
        axes[i,0].set_xlabel("Theoretical Cumulative Probability")
        axes[i,0].set_ylabel("Empirical Cumulative Probability")
        axes[i,0].grid(alpha=0.3)
        axes[i,0].legend()
        
    plt.tight_layout()
    save_path = f"analysis_result/{file_prefix}_pp_qq_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_candidate_failure_vs_empirical(data, h_hat, dist_results, file_suffix: str):
    """绘制所有候选Distribution的失效率曲线与经验失效率曲线"""
    n_dist = len(dist_results)
    fig, axes = plt.subplots(n_dist, figsize=(10, 2 * n_dist))
    fig.suptitle("Empirical and Candidate Failure Rate Curves", fontsize=16, y=0.98)
    
    data_sorted = np.sort(data)
    for i, res in enumerate(dist_results):
        dist = res["dist_obj"]
        dist_name = res["dist_en"]
        
        # 生成平滑 x 轴让理论失效率曲线光滑
        x_plot = np.linspace(data_sorted.min(), data_sorted.max(), 200)
        # 计算理论失效率
        pdf = dist.pdf(x_plot)
        surv = 1 - dist.cdf(x_plot)
        surv = np.clip(surv, 1e-10, 1.0)  # 防止除零
        h_theory = pdf / surv
        
        # 理论失效率
        axes[i].plot(x_plot, h_theory, 'r-', lw=2, label=f'Theoretical Failure {dist_name}')
        # 经验失效率
        axes[i].plot(data_sorted, h_hat, 'bo', ms=3, alpha=0.6, label="Empirical Failure Rate")
        
        # 子图样式
        axes[i].set_title(f'Empirical vs Theoretical Failure - {dist_name}')
        axes[i].set_xlabel('Failure Time t')
        axes[i].set_ylabel('Failure Rate h(t)')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(f"analysis_result/hazard_rate_comparison_{file_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def fit_and_evaluate_distributions(data, h_hat,
                                   pp_qq_file_prefix, failure_vs_empirical_file_suffix,
                                   print_table=True, plot_distributions=True
                                   ):
    '''
    分析所有候选单一Distribution
    单一Distribution接受准则:
    1. AD 检验的 p 值 > 0.05
    2. Q-Q 图上的点基本落在直线上, 特别是尾部点
    3. 风险率曲线形状与经验风险率曲线一致
    '''
    results = []
    for en_name in DISTRIBUTIONS.keys():
        res = analyze_distribution(data, en_name)
        results.append(res)
        
    # 可视化Distribution
    if plot_distributions:
        plot_pp_qq(data, results, pp_qq_file_prefix)
        plot_candidate_failure_vs_empirical(data, h_hat, results, failure_vs_empirical_file_suffix)
    
    # 输出表格结果
    if print_table:
        print("="*120)
        print(" " * 28 + "MLE Fit + AD Test + AIC/BIC Results")
        print("="*120)
        # 表头
        print(f"{'Distribution':<16} {'Params':<35} {'Statistic':^12} {'P-value':^10} {'AIC':^14} {'BIC':^14}")
        print("-"*120)
        
        # 按 AIC 升序排序
        results_sorted = sorted(results, key=lambda x: x["aic"])
        for r in results_sorted:
            stat = r["ad_stat"]
            test_label = "[AD]"
            
            # Distribution名 + 检验标签
            dist_name = f"{r['zh_name']} {test_label}".ljust(16)
            # Params格式化
            params_str = "[" + " ".join(f"{v:8.4f}" for v in r["params"]) + "]"
            # 打印
            print(
                f"{dist_name}"
                f"{params_str:<35} "
                f"{stat:>10.4f}        "
                f"{r['pvalue']:>8.4f}    "
                f"{r['aic']:>13.4f}   "
                f"{r['bic']:>13.4f}"
            )
    
'''======== 多失效Mechanism无监督拟合 ========'''
class MixtureDistributionEM:
    """不同类型混合Distribution的 EM 算法"""
    # Distribution注册中心: 名称 -> (pdf函数, Params个数, 初始化函数, Params边界)
    _DISTRIBUTIONS = {
        "expon": (
            lambda t, *params: expon.pdf(t, loc=params[0], scale=params[1]),
            2,
            lambda data: (0.0, np.mean(data)),  # (loc, scale)
            [(None, None), (1e-3, None)]
        ),
        "weibull_min": (
            lambda t, *params: weibull_min.pdf(t, c=params[0], loc=0, scale=params[1]),
            2,
            lambda data: (1.0, np.mean(data)),  # (c=k, scale=lam)
            [(1e-3, None), (1e-3, None)]
        ),
        "norm": (
            lambda t, *params: norm.pdf(t, loc=params[0], scale=params[1]),
            2,
            lambda data: (np.mean(data), np.std(data)),  # (mu, sigma)
            [(None, None), (1e-3, None)]
        ),
        "lognorm": (
            lambda t, *params: lognorm.pdf(t, s=params[0], loc=0, scale=params[1]),
            2,
            lambda data: (1.0, np.exp(np.mean(np.log(data)))),  # (s, scale)
            [(1e-3, None), (1e-3, None)]
        ),
        "gamma": (
            lambda t, *params: gamma.pdf(t, a=params[0], loc=0, scale=params[1]),
            2,
            lambda data: (1.0, np.mean(data)),  # (a, scale)
            [(1e-3, None), (1e-3, None)]
        )
    }
    
    def __init__(self, data, dist_names, tol=1e-6, max_iter=2000):
        self.data = np.asarray(data).flatten()  # 确保失效寿命数据 (n,) 一维
        self.dist_names = dist_names            # 每个分量的Distribution类型
        self.K = len(dist_names)                # 分量数 = Distribution列表长度
        self.tol = tol                          # 收敛阈值
        self.max_iter = max_iter
        self.n = len(self.data)
        
        # 校验 K 值合法性
        if self.K <= 1:
            raise ValueError(f"混合Distribution要求分量数量 K ≥ 2, 当前传入分量数: {self.K}, 无法构建混合Distribution")
        if self.K > 4:
            import warnings
            warnings.warn(
                f"当前分量数量 K = {self.K}, 超过推荐值 4\n"
                "过多分量会导致模型过拟合、Params难以收敛、可解释性大幅下降, 建议 K ≤ 4",
                UserWarning
            )
        
        # 每个分量独立配置
        self.pdf_funcs = []       # 每个分量的 pdf 函数
        self.n_params_list = []   # 每个分量的Params个数
        self.init_funcs = []      # 每个分量的初始化函数
        self.bounds_list = []     # 每个分量的Params边界
        self._rebuild_distribution_runtime()
            
        # 初始化权重 π, Params
        self.pi = np.ones(self.K) / self.K
        self.params = [init(self.data) for init in self.init_funcs]  # list 格式
        self.log_likelihoods = []

    def _rebuild_distribution_runtime(self):
        """根据Distribution名称重建运行期函数表。

        这些 PDF 和初始化函数来自类级注册表，包含 lambda 对象。lambda 不能被
        pickle 稳定序列化，因此实例持久化时不能直接保存 self.pdf_funcs 和
        self.init_funcs。模型加载后通过 dist_names 重新构建这些运行期字段即可。
        """
        self.pdf_funcs = []
        self.n_params_list = []
        self.init_funcs = []
        self.bounds_list = []
        for name in self.dist_names:
            if name not in self._DISTRIBUTIONS:
                raise ValueError(f"仅支持：{list(self._DISTRIBUTIONS.keys())}")
            pdf, n_p, init, bnd = self._DISTRIBUTIONS[name]
            self.pdf_funcs.append(pdf)
            self.n_params_list.append(n_p)
            self.init_funcs.append(init)
            self.bounds_list.append(bnd)

    def __getstate__(self):
        """返回可 pickle 的模型状态。

        混合分布模型的真实可复现信息是数据、分布名称、权重、参数和似然轨迹；
        PDF 函数、初始化函数和边界可以由 dist_names 重新生成，因此这里主动移除，
        避免 pickle 匿名函数导致训练流程中断。
        """
        state = self.__dict__.copy()
        state.pop("pdf_funcs", None)
        state.pop("init_funcs", None)
        state.pop("bounds_list", None)
        state.pop("n_params_list", None)
        return state

    def __setstate__(self, state):
        """恢复 pickle 模型后重建运行期函数表。"""
        self.__dict__.update(state)
        self._rebuild_distribution_runtime()
        
    def _initialize_with_kmeans(self):
        """K-means 初始化"""
        # K-means 聚类
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(self.data.reshape(-1, 1))
        
        # 按聚类结果初始化各成分Params
        for k in range(self.K):
            cluster = self.data[labels == k]
            if len(cluster) == 0:
                self.params[k] = self.init_funcs[k](self.data)
            else:
                self.params[k] = self.init_funcs[k](cluster)

    def _pdf(self, t, component_idx):
        """第 k 个分量使用自己的 PDF 和Params"""
        return self.pdf_funcs[component_idx](t, *self.params[component_idx])
    
    def _e_step(self):
        """E 步: 计算后验概率 gamma"""
        pdf_vals = np.zeros((self.n, self.K))
        for k in range(self.K):
            # 计算原始 PDF
            raw_pdf = self._pdf(self.data, k)
            
            # 替换为极小值
            raw_pdf = np.nan_to_num(raw_pdf, nan=1e-12, posinf=1e-12, neginf=1e-12)
            
            pdf_vals[:, k] = self.pi[k] * raw_pdf
            
        # 后验概率
        total_pdf = np.sum(pdf_vals, axis=1, keepdims=True)
        total_pdf[total_pdf == 0] = 1e-10  # 避免除零
        gamma = pdf_vals / total_pdf
        
        return gamma, pdf_vals
    
    def _m_step(self, gamma):
        """更新Params π, k, λ"""
        # 更新混合系数 pi
        Nk = np.sum(gamma, axis=0)
        self.pi = Nk / self.n
        
        # 对每个分量独立优化DistributionParams
        for k in range(self.K):
            if Nk[k] < 1e-10:
                continue
            weights = gamma[:, k]
            x0 = self.params[k]
            bounds_k = self.bounds_list[k]  # 分量 k 的边界
            
            # 负加权对数似然
            def neg_log_lik(params):
                """执行 neg log lik 对应的项目处理逻辑。"""
                pdf = self.pdf_funcs[k](self.data, *params)
                pdf = np.clip(pdf, 1e-10, None)
                return -np.sum(weights * np.log(pdf))
            
            # 优化求解
            res = minimize(neg_log_lik, x0=x0, bounds=bounds_k, method="L-BFGS-B")
            self.params[k] = res.x
            
    def fit(self):
        """训练 EM 算法"""
        self._initialize_with_kmeans()
        prev_log_likelihoods = -np.inf
        
        for _ in range(self.max_iter):
            check_cancelled("混合分布 EM 拟合")
            # E步
            gamma, pdf_vals = self._e_step()
            # 计算对数似然
            log_likelihoods = np.sum(np.log(np.sum(pdf_vals, axis=1) + 1e-10))
            self.log_likelihoods.append(log_likelihoods)
            
            # 收敛判断
            if abs(log_likelihoods - prev_log_likelihoods) < self.tol:
                break
            prev_log_likelihoods = log_likelihoods
            
            # M步
            self._m_step(gamma)
        return self
    
    def get_posterior(self):
        """获取所有样本的后验概率 p_ik"""
        gamma, _ = self._e_step()
        return gamma
    
    def get_bic(self):
        """计算 BIC 值"""
        log_likelihoods = self.log_likelihoods[-1] if self.log_likelihoods else -np.inf
        n_params = (self.K - 1) + sum(self.n_params_list)  # 计算总Params个数
        bic = -2 * log_likelihoods + n_params * np.log(self.n)
        return bic
    
    def get_params(self):
        """返回最终Params: pi + 各分量[Distribution名, Params]"""
        return self.pi.copy(), list(zip(self.dist_names, self.params))
    
def generate_dist_combinations(K, supported_dists, mode="default"):
    """
    生成 K 个分量的Distribution组合
    mode:
        - "default"    : 可重复、无顺序
        - "ordered"    : 可重复、有顺序
        - "only_same"  : 仅自身重复
    """
    from itertools import combinations_with_replacement, product
    
    if mode == "ordered":
        return list(product(supported_dists, repeat=K))
    elif mode == "only_same":
        return [(d,) * K for d in supported_dists]
    else:
        return list(combinations_with_replacement(supported_dists, K))

def select_best_mixture(data, max_k=4):
    """遍历 K=2~4 所有Distribution组合, 用 BIC 选择最优混合Distribution模型"""
    supported_dists = list(MixtureDistributionEM._DISTRIBUTIONS.keys())
    
    all_models = []       # 存储所有训练好的模型
    all_bics = []         # 存储所有模型的 BIC
    all_dist_combos = []  # 存储每个模型对应的Distribution组合
    all_K = []            # 存储每个模型对应的 K
    
    # 遍历分量数 K=2 ~ max_k
    for K in range(2, max_k + 1):
        check_cancelled("混合分布模型选择")
        # 生成当前 K 下所有可能的Distribution组合
        dist_combinations = generate_dist_combinations(K, supported_dists)
        
        for dist_names in dist_combinations:
            check_cancelled("混合分布候选组合拟合")
            try:
                model = MixtureDistributionEM(data, dist_names=list(dist_names)).fit()
                
                # 记录结果
                all_models.append(model)
                all_bics.append(model.get_bic())
                all_dist_combos.append(dist_names)
                all_K.append(K)
            except TaskCancelledError:
                raise
            except Exception as e:
                print(f"[fit_failed] K = {K}, distributions = {dist_names}, error: {str(e)}")
                continue
    
    # 找到 BIC 最小的最优模型
    if not all_bics:
        raise RuntimeError("所有模型训练失败, 请检查数据或Distribution支持情况")
    
    best_idx = np.argmin(all_bics)
    best_model = all_models[best_idx]
    best_bic = all_bics[best_idx]
    best_dist_combo = all_dist_combos[best_idx]
    best_k = all_K[best_idx]
    
    return best_model, best_bic, best_dist_combo, best_k

def classify_samples(posterior, threshold=0.7):
    """分配到后验概率最大的Mechanism并标记最大后验概率 < 阈值的样本为模糊样本"""
    max_proba = np.max(posterior, axis=1)
    labels = np.argmax(posterior, axis=1)
    fuzzy_mask = max_proba < threshold
    return labels, fuzzy_mask
    
def plot_failure_classification_and_hazard(data, h_hat, labels, fuzzy_mask, best_k):
    '''绘制失效样本分类与经验失效率曲线'''
    plt.figure(figsize=(12, 5))
    
    # 寿命数据Distribution
    plt.subplot(121)
    for k in range(best_k):
        mask = labels == k
        plt.hist(data[mask], bins=20, alpha=0.5, label=f'Mechanism {k+1}')
    plt.hist(data[fuzzy_mask], bins=10, alpha=0.3, color='gray', label='Fuzzy Samples')
    plt.xlabel('Failure Lifetime')
    plt.ylabel('Frequency')
    plt.title(f'Failure Sample Classification (K={best_k})')
    plt.legend()
    
    # 经验失效率曲线
    plt.subplot(122)
    plt.plot(data, h_hat)
    plt.xlabel('Lifetime')
    plt.ylabel('Empirical Hazard Rate')
    plt.title('Empirical Hazard Rate Curve')
    
    plt.tight_layout()
    plt.savefig('analysis_result/failure_classification_hazard_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def failure_mechanism_analysis(data, h_hat, print_results=True, save_plots=True, save_model=True):
    """完整失效Mechanism分析流程"""
    # BIC 准则选择最优混合Distribution
    best_model, best_bic, best_dist_combo, best_k = select_best_mixture(data)
    pi, params = best_model.get_params()
    
    # 计算后验概率 + 样本分类
    posterior = best_model.get_posterior()
    labels, fuzzy_mask = classify_samples(posterior)
    
    # 统计结果
    if print_results:
        print("\n===== Best Mixture Distribution Result =====")
        print(f"BIC = {best_bic:.2f}")
        print(f"Best component count K = {best_k}")
        print(f"Best distribution combination = {best_dist_combo}")
        print(f"Mixture weights pi = {pi.round(4)}")
        print(f"Component parameters = {params}")
        
        n_fuzzy = np.sum(fuzzy_mask)
        print(f"\n===== Sample Classification Result =====")
        print(f"Total samples: {len(data)}")
        print(f"Fuzzy samples: {n_fuzzy} ({n_fuzzy/len(data)*100:.2f}%)")
        for k in range(best_k):
            n_k = np.sum(labels == k)
            print(f"Mechanism {k+1} samples: {n_k}")
        
    # 绘制失效样本分类与经验失效率曲线
    if save_plots:
        plot_failure_classification_and_hazard(data, h_hat, labels, fuzzy_mask, best_k)
        
    if save_model:
        # 自动创建保存目录
        save_dir = "analysis_result/mixture_models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        dist_str = "_".join(best_dist_combo)
        bic_str = f"bic{best_bic:.0f}"
        
        # 保存路径
        model_path = os.path.join(
            save_dir,
            f"failure_mixture_model_{dist_str}_{bic_str}.pkl"
        )
        
        # 保存模型
        model_params = {
            # 模型结构Params
            "k": best_k,                    # 分量数量
            "dist_combo": best_dist_combo,  # Best distribution combination
            "pi": pi,                       # 混合权重
            "params": params,               # Component parameters
            
            # 核心使用数据
            "posterior": posterior,         # 所有样本后验概率矩阵 (n_samples, K)
            "labels": labels,               # 样本分类标签
            "fuzzy_mask": fuzzy_mask,       # 模糊样本掩码
        }
        
        # 保存Params字典
        joblib.dump(model_params, model_path)
        print(f"\nModel saved to: {model_path}")
        
    return model_params

'''======== 多失效Mechanism混合Distribution模型评估 ========'''
def calculate_edf_fit_goodness(model_params, sorted_data):
    """计算 AD 和 KS 拟合优度Statistic"""
    # 计算模型的累积Distribution函数值
    F_model = np.zeros(len(sorted_data), dtype=np.float64)
    pi = model_params['pi']
    params_list = model_params['params']
    
    for i, t in enumerate(sorted_data):
        cdf_val = 0.0
        for k in range(model_params['k']):
            dist_name, params = params_list[k]
            if dist_name == "expon":
                cdf_val += pi[k] * expon.cdf(t, loc=params[0], scale=params[1])
            elif dist_name == "weibull_min":
                cdf_val += pi[k] * weibull_min.cdf(t, c=params[0], loc=0, scale=params[1])
            elif dist_name == "norm":
                cdf_val += pi[k] * norm.cdf(t, loc=params[0], scale=params[1])
            elif dist_name == "lognorm":
                cdf_val += pi[k] * lognorm.cdf(t, s=params[0], loc=0, scale=params[1])
            elif dist_name == "gamma":
                cdf_val += pi[k] * gamma.cdf(t, a=params[0], loc=0, scale=params[1])
        F_model[i] = cdf_val
    
    # 中位秩法计算经验Distribution函数
    n = len(sorted_data)
    F_emp = (np.arange(1, n+1) - 0.3) / (n + 0.4)
    
    # KS 检验
    D, p_value_ks = stats.kstest(F_model, 'uniform')
    
    # Anderson-Darling 检验
    eps = 1e-10
    F_model = np.clip(F_model, eps, 1 - eps)
    A2 = -n - np.sum((2*np.arange(1, n+1)-1) * (np.log(F_model) + np.log(1 - F_model[::-1]))) / n
    
    return {
        "AD_statistic": A2,
        "KS_statistic": D,
        "KS_p_value": p_value_ks,
        "F_model": F_model,
        "F_emp": F_emp
    }
    
def plot_fit_comparison(sorted_data, model_params, fit_results):
    """绘制P-P图、Q-Q图、可靠度曲线对比和PDF对比"""
    F_emp = fit_results["F_emp"]
    F_model = fit_results["F_model"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # P-P 图
    axes[0,0].plot(F_emp, F_model, 'o', markersize=4)
    axes[0,0].plot([0,1], [0,1], 'r--')
    axes[0,0].set_xlabel('Empirical CDF')
    axes[0,0].set_ylabel('Model CDF')
    axes[0,0].set_title('P-P Plot')
    axes[0,0].grid(True, alpha=0.3)
    
    # Q-Q 图
    # 计算模型分位数
    q_emp = np.percentile(sorted_data, np.linspace(1, 99, 99))
    q_model = np.zeros_like(q_emp)
    pi = model_params['pi']
    params_list = model_params['params']
    
    # 简单的分位数计算
    t_grid = np.linspace(0, np.max(sorted_data)*1.2, 10000)
    F_grid = np.zeros_like(t_grid)
    for k in range(model_params['k']):
        dist_name, params = params_list[k]
        if dist_name == "expon":
            F_grid += pi[k] * expon.cdf(t_grid, loc=params[0], scale=params[1])
        elif dist_name == "weibull_min":
            F_grid += pi[k] * weibull_min.cdf(t_grid, c=params[0], loc=0, scale=params[1])
        elif dist_name == "norm":
            F_grid += pi[k] * norm.cdf(t_grid, loc=params[0], scale=params[1])
        elif dist_name == "lognorm":
            F_grid += pi[k] * lognorm.cdf(t_grid, s=params[0], loc=0, scale=params[1])
        elif dist_name == "gamma":
            F_grid += pi[k] * gamma.cdf(t_grid, a=params[0], loc=0, scale=params[1])
    
    for i, p in enumerate(np.linspace(0.01, 0.99, 99)):
        q_model[i] = t_grid[np.argmin(np.abs(F_grid - p))]
    
    axes[0,1].plot(q_emp, q_model, 'o', markersize=4)
    axes[0,1].plot([np.min(q_emp), np.max(q_emp)], [np.min(q_emp), np.max(q_emp)], 'r--')
    axes[0,1].set_xlabel('Empirical Quantiles')
    axes[0,1].set_ylabel('Model Quantiles')
    axes[0,1].set_title('Q-Q Plot')
    axes[0,1].grid(True, alpha=0.3)
    
    # 可靠度曲线对比
    R_emp = 1 - F_emp
    R_model = 1 - F_model
    axes[1,0].plot(sorted_data, R_emp, 'b-', label='Empirical')
    axes[1,0].plot(sorted_data, R_model, 'r--', label='Model')
    axes[1,0].set_xlabel('Lifetime')
    axes[1,0].set_ylabel('Reliability')
    axes[1,0].set_title('Reliability Curve Comparison')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 核密度估计与模型 PDF 对比
    kde = gaussian_kde(sorted_data)
    t_pdf = np.linspace(0, np.max(sorted_data), 200)
    pdf_kde = kde(t_pdf)
    
    pdf_model = np.zeros_like(t_pdf)
    for k in range(model_params['k']):
        dist_name, params = params_list[k]
        if dist_name == "expon":
            pdf_model += pi[k] * expon.pdf(t_pdf, loc=params[0], scale=params[1])
        elif dist_name == "weibull_min":
            pdf_model += pi[k] * weibull_min.pdf(t_pdf, c=params[0], loc=0, scale=params[1])
        elif dist_name == "norm":
            pdf_model += pi[k] * norm.pdf(t_pdf, loc=params[0], scale=params[1])
        elif dist_name == "lognorm":
            pdf_model += pi[k] * lognorm.pdf(t_pdf, s=params[0], loc=0, scale=params[1])
        elif dist_name == "gamma":
            pdf_model += pi[k] * gamma.pdf(t_pdf, a=params[0], loc=0, scale=params[1])
    
    axes[1,1].plot(t_pdf, pdf_kde, 'b-', label='KDE (Empirical)')
    axes[1,1].plot(t_pdf, pdf_model, 'r--', label='Model')
    axes[1,1].set_xlabel('Lifetime')
    axes[1,1].set_ylabel('Probability Density')
    axes[1,1].set_title('PDF Comparison')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_result/fit_goodness_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def calculate_key_life_metrics(model_params, sorted_data):
    """计算关键寿命Metric并对比Empirical与Model prediction值"""
    # Empirical计算
    metrics_emp = {
        "MTTF": np.mean(sorted_data),
        "t_B1": np.percentile(sorted_data, 1),
        "t_B10": np.percentile(sorted_data, 10),
        "t_median": np.median(sorted_data),
        "t_B90": np.percentile(sorted_data, 90),
        "t_B99": np.percentile(sorted_data, 99)
    }
    
    # Model prediction值计算
    t_grid = np.linspace(0, np.max(sorted_data)*1.2, 10000)
    F_grid = np.zeros_like(t_grid)
    pi = model_params['pi']
    params_list = model_params['params']
    
    for k in range(model_params['k']):
        dist_name, params = params_list[k]
        if dist_name == "expon":
            F_grid += pi[k] * expon.cdf(t_grid, loc=params[0], scale=params[1])
        elif dist_name == "weibull_min":
            F_grid += pi[k] * weibull_min.cdf(t_grid, c=params[0], loc=0, scale=params[1])
        elif dist_name == "norm":
            F_grid += pi[k] * norm.cdf(t_grid, loc=params[0], scale=params[1])
        elif dist_name == "lognorm":
            F_grid += pi[k] * lognorm.cdf(t_grid, s=params[0], loc=0, scale=params[1])
        elif dist_name == "gamma":
            F_grid += pi[k] * gamma.cdf(t_grid, a=params[0], loc=0, scale=params[1])
    
    def get_quantile(p):
        """执行 get quantile 对应的项目处理逻辑。"""
        return t_grid[np.argmin(np.abs(F_grid - p))]
    
    metrics_model = {
        "MTTF": np.trapezoid(1-F_grid, t_grid),  # 积分计算平均寿命
        "t_B1": get_quantile(0.01),
        "t_B10": get_quantile(0.1),
        "t_median": get_quantile(0.5),
        "t_B90": get_quantile(0.90),
        "t_B99": get_quantile(0.99)
    }
    
    # 计算相对误差
    errors = {}
    for key in metrics_emp:
        errors[key] = abs(metrics_model[key] - metrics_emp[key]) / metrics_emp[key] * 100
    
    return {
        "empirical": metrics_emp,
        "model": metrics_model,
        "relative_error_percent": errors
    }
    
def plot_hazard_comparison(model_params, sorted_data, h_hat):
    """绘制经验风险率与模型风险率对比图"""
    sorted_data = np.sort(sorted_data)
    t_grid = np.linspace(0, np.max(sorted_data), 200)
    
    # 计算模型风险率
    pi = model_params['pi']
    params_list = model_params['params']
    pdf_model = np.zeros_like(t_grid)
    cdf_model = np.zeros_like(t_grid)
    
    for k in range(model_params['k']):
        dist_name, params = params_list[k]
        if dist_name == "expon":
            pdf_model += pi[k] * expon.pdf(t_grid, loc=params[0], scale=params[1])
            cdf_model += pi[k] * expon.cdf(t_grid, loc=params[0], scale=params[1])
        elif dist_name == "weibull_min":
            pdf_model += pi[k] * weibull_min.pdf(t_grid, c=params[0], loc=0, scale=params[1])
            cdf_model += pi[k] * weibull_min.cdf(t_grid, c=params[0], loc=0, scale=params[1])
        elif dist_name == "norm":
            pdf_model += pi[k] * norm.pdf(t_grid, loc=params[0], scale=params[1])
            cdf_model += pi[k] * norm.cdf(t_grid, loc=params[0], scale=params[1])
        elif dist_name == "lognorm":
            pdf_model += pi[k] * lognorm.pdf(t_grid, s=params[0], loc=0, scale=params[1])
            cdf_model += pi[k] * lognorm.cdf(t_grid, s=params[0], loc=0, scale=params[1])
        elif dist_name == "gamma":
            pdf_model += pi[k] * gamma.pdf(t_grid, a=params[0], loc=0, scale=params[1])
            cdf_model += pi[k] * gamma.cdf(t_grid, a=params[0], loc=0, scale=params[1])
    
    h_model = pdf_model / (1 - cdf_model + 1e-10)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_data, h_hat, 'b-', label='Empirical Hazard Rate')
    plt.plot(t_grid, h_model, 'r--', label='Model Hazard Rate')
    plt.xlabel('Lifetime')
    plt.ylabel('Hazard Rate')
    plt.title('Hazard Rate Function Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('analysis_result/hazard_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def bootstrap_uncertainty_analysis(model_params, sorted_data, n_bootstrap=1000):
    """使用Bootstrap法评估关键Metric的不确定性"""
    np.random.seed(42)  # 固定随机种子保证可复现
    
    bootstrap_results = {
        "B1": [],
        "B10": [],
        "B90": [],
        "B99": [],
        "MTTF": [],
        "pi": [],
        "params": []
    }
    
    for i in range(n_bootstrap):
        check_cancelled("混合分布 Bootstrap 分析")
        # 有放回重采样
        bootstrap_data = np.random.choice(sorted_data, size=len(sorted_data), replace=True)
        
        try:
            # 重新拟合模型
            bootstrap_model = MixtureDistributionEM(bootstrap_data, dist_names=model_params['dist_combo']).fit()
            
            # 计算关键Metric
            t_grid = np.linspace(0, np.max(bootstrap_data)*1.2, 10000)
            F_grid = np.zeros_like(t_grid)
            pi, params_list = bootstrap_model.get_params()
            
            for k in range(bootstrap_model.K):
                dist_name, params = params_list[k]
                if dist_name == "expon":
                    F_grid += pi[k] * expon.cdf(t_grid, loc=params[0], scale=params[1])
                elif dist_name == "weibull_min":
                    F_grid += pi[k] * weibull_min.cdf(t_grid, c=params[0], loc=0, scale=params[1])
                elif dist_name == "norm":
                    F_grid += pi[k] * norm.cdf(t_grid, loc=params[0], scale=params[1])
                elif dist_name == "lognorm":
                    F_grid += pi[k] * lognorm.cdf(t_grid, s=params[0], loc=0, scale=params[1])
                elif dist_name == "gamma":
                    F_grid += pi[k] * gamma.cdf(t_grid, a=params[0], loc=0, scale=params[1])
            
            def get_quantile(p):
                """执行 get quantile 对应的项目处理逻辑。"""
                return t_grid[np.argmin(np.abs(F_grid - p))]
            
            bootstrap_results["B1"].append(get_quantile(0.01))
            bootstrap_results["B10"].append(get_quantile(0.1))
            bootstrap_results["B90"].append(get_quantile(0.9))
            bootstrap_results["B99"].append(get_quantile(0.99))
            bootstrap_results["MTTF"].append(np.trapezoid(1-F_grid, t_grid))
            bootstrap_results["pi"].append(pi)
            bootstrap_results["params"].append(params_list)
            
        except TaskCancelledError:
            raise
        except Exception as e:
            print(f"Bootstrap iteration {i} failed: {str(e)}")
            continue
    
    # 计算 95% 置信区间
    confidence_intervals = {}
    for key in ["MTTF", "B1", "B10", "B90", "B99"]:
        values = np.array(bootstrap_results[key])
        confidence_intervals[key] = {
            "mean": np.mean(values),
            "lower": np.percentile(values, 2.5),
            "upper": np.percentile(values, 97.5),
            "std": np.std(values)
        }
    
    return confidence_intervals

def cross_validation_analysis(sorted_data, dist_names, n_splits=5):
    """k折交叉验证评估Model prediction能力"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_list = []
    mae_list = []
    
    for train_idx, test_idx in kf.split(sorted_data):
        check_cancelled("混合分布交叉验证")
        train_data = sorted_data[train_idx]
        test_data = sorted_data[test_idx]
        
        try:
            # 在训练集上拟合模型
            model = MixtureDistributionEM(train_data, dist_names=dist_names).fit()
            
            # 计算测试集的预测误差
            sorted_test = np.sort(test_data)
            F_emp_test = (np.arange(1, len(sorted_test)+1) - 0.3) / (len(sorted_test) + 0.4)
            
            # 计算模型在测试点的 CDF
            F_model_test = np.zeros_like(sorted_test)
            pi, params_list = model.get_params()
            
            for i, t in enumerate(sorted_test):
                cdf_val = 0.0
                for k in range(model.K):
                    dist_name, params = params_list[k]
                    if dist_name == "expon":
                        cdf_val += pi[k] * expon.cdf(t, loc=params[0], scale=params[1])
                    elif dist_name == "weibull_min":
                        cdf_val += pi[k] * weibull_min.cdf(t, c=params[0], loc=0, scale=params[1])
                    elif dist_name == "norm":
                        cdf_val += pi[k] * norm.cdf(t, loc=params[0], scale=params[1])
                    elif dist_name == "lognorm":
                        cdf_val += pi[k] * lognorm.cdf(t, s=params[0], loc=0, scale=params[1])
                    elif dist_name == "gamma":
                        cdf_val += pi[k] * gamma.cdf(t, a=params[0], loc=0, scale=params[1])
                F_model_test[i] = cdf_val
            
            # 计算误差
            mse = np.mean((F_model_test - F_emp_test)**2)
            mae = np.mean(np.abs(F_model_test - F_emp_test))
            
            mse_list.append(mse)
            mae_list.append(mae)
            
        except TaskCancelledError:
            raise
        except Exception as e:
            print(f"Cross-validation fold failed: {str(e)}")
            continue
        
    return {
        "mean_mse": np.mean(mse_list),
        "std_mse": np.std(mse_list),
        "mean_mae": np.mean(mae_list),
        "std_mae": np.std(mae_list)
    }
    
def complete_model_evaluation(sorted_data, h_hat, model_params):
    """完整的混合Distribution模型评估流程"""
    print("\n" + "="*50)
    print("Running full mixture-model evaluation")
    print("="*50)
    
    # Statistical goodness-of-fit evaluation
    fit_results = calculate_edf_fit_goodness(model_params, sorted_data)
    
    print("\nStatistical goodness-of-fit evaluation")
    print(f"AD statistic: {fit_results['AD_statistic']:.4f}")
    print(f"KS statistic: {fit_results['KS_statistic']:.4f}")
    print(f"KS test p-value: {fit_results['KS_p_value']:.4f}")
    
    # 绘制拟合对比图
    plot_fit_comparison(sorted_data, model_params, fit_results)
    
    # Engineering applicability evaluation
    print("\nEngineering applicability evaluation")
    life_metrics = calculate_key_life_metrics(model_params, sorted_data)
    
    print("\nKey lifetime metric comparison:")
    print(f"{'Metric':<10} {'Empirical':<10} {'Model prediction':<10} {'Relative error (%)':<10}")
    print("-"*40)
    for key in life_metrics["empirical"]:
        emp = life_metrics["empirical"][key]
        mod = life_metrics["model"][key]
        err = life_metrics["relative_error_percent"][key]
        print(f"{key:<10} {emp:<10.2f} {mod:<10.2f} {err:<10.2f}")
    
    # 绘制风险率对比图
    plot_hazard_comparison(model_params, sorted_data, h_hat)
    
    # Model robustness and uncertainty evaluation
    print("\nModel robustness and uncertainty evaluation")
    print("Running bootstrap analysis...")
    ci_results = bootstrap_uncertainty_analysis(model_params, sorted_data, n_bootstrap=1000)
    
    print("\nKey metric 95% confidence intervals:")
    print(f"{'Metric':<10} {'Mean':<10} {'Lower':<10} {'Upper':<10} {'CV (%)':<10}")
    print("-"*60)
    for key in ci_results:
        mean_val = ci_results[key]["mean"]
        lower = ci_results[key]["lower"]
        upper = ci_results[key]["upper"]
        cv = ci_results[key]["std"] / mean_val * 100
        print(f"{key:<10} {mean_val:<10.2f} {lower:<10.2f} {upper:<10.2f} {cv:<10.2f}")
    
    # Model predictive ability evaluation
    print("\nModel predictive ability evaluation")
    cv_results = cross_validation_analysis(sorted_data, model_params['dist_combo'])
    
    print(f"5-fold CV mean MSE: {cv_results['mean_mse']:.6f}")
    print(f"5-fold CV mean MAE: {cv_results['mean_mae']:.6f}")
    
    # Overall evaluation conclusion
    print("\n" + "="*50)
    print("Overall evaluation conclusion")
    print("="*50)
    
    # 自动生成评估结论
    conclusion = []
    if fit_results["AD_statistic"] < 1.0 and fit_results["KS_statistic"] < 0.2:
        conclusion.append("Good statistical fit")
    elif fit_results["AD_statistic"] < 2.0 and fit_results["KS_statistic"] < 0.3:
        conclusion.append("Moderate statistical fit, acceptable")
    else:
        conclusion.append("Poor statistical fit, unacceptable")
    
    if life_metrics["relative_error_percent"]["t_B10"] < 10:
        conclusion.append("B10 prediction error is acceptable")
    elif life_metrics["relative_error_percent"]["t_B10"] < 20:
        conclusion.append("B10 prediction error is high; use cautiously")
    else:
        conclusion.append("B10 prediction error is too high; model unacceptable")
    
    if ci_results["B10"]["std"] / ci_results["B10"]["mean"] < 0.5:
        conclusion.append("Model uncertainty is low")
    else:
        conclusion.append("Model uncertainty is high; consider more samples")
    
    for line in conclusion:
        print(line)
    



# =============================================================================
# Project pipeline API
# =============================================================================
from pathlib import Path as _Path
from typing import Any as _Any, Dict as _Dict, Mapping as _Mapping
import pandas as _pd


def extract_lifetimes_from_train_df(train_df: _pd.DataFrame) -> np.ndarray:
    """从训练集提取每台发动机的失效寿命。"""
    from lvd_surv.data.cmapss import normalize_cmapss_schema

    df = normalize_cmapss_schema(train_df, add_condition_from_ops=True)
    if "failure_time" in df.columns:
        values = df.groupby("unit_id")["failure_time"].max().to_numpy(dtype=float)
    else:
        values = df.groupby("unit_id")["cycle"].max().to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        raise ValueError("No positive lifetimes are available for mixture-prior fitting.")
    return values


def fit_lifetime_mixture_prior(
    train_df: _pd.DataFrame,
    max_k: int = 4,
    dataset: str = "unknown",
) -> _Dict[str, _Any]:
    """拟合寿命混合Distribution并返回主流程可消费的先验契约"""
    from lvd_surv.core.contracts import mixture_prior_contract_from_user_result

    lifetimes = extract_lifetimes_from_train_df(train_df)
    lifetimes = iqr_outlier_detection(lifetimes, verbose=False)
    raw_result = select_best_mixture(lifetimes, max_k=int(max_k))
    contract = mixture_prior_contract_from_user_result(raw_result, dataset=str(dataset))
    if contract is None:
        raise ValueError("select_best_mixture returned an unsupported result shape.")
    contract.setdefault("metadata", {})
    contract["metadata"].update({"num_lifetimes": int(len(lifetimes)), "max_k": int(max_k), "algorithm": "original_em_mixture"})
    return {"mixture_prior": contract, "raw_model": raw_result, "lifetimes": lifetimes}


def get_or_build_lifetime_prior(cfg: _Mapping[str, _Any], train_df: _pd.DataFrame, paths: _Mapping[str, _Path]) -> _Dict[str, _Any]:
    """读取或生成寿命混合Distribution先验缓存"""
    from lvd_surv.core.artifacts import (
        build_config_fingerprint,
        build_data_fingerprint,
        build_manifest,
        build_script_fingerprint,
        get_dataset_name,
        is_cache_valid,
        load_json_artifact,
        save_json_artifact,
        save_pickle_artifact,
        update_run_state,
    )

    prior_cfg = dict(cfg.get("lifetime_prior", {}))
    if not bool(prior_cfg.get("enabled", True)):
        return {"mixture_prior": None, "raw_model": None, "cache_hit": False}
    dataset = get_dataset_name(cfg)
    policy = str(prior_cfg.get("cache_policy", "auto")).lower()
    cols = [c for c in ["unit_id", "cycle", "rul", "RUL", "failure_time"] if c in train_df.columns]
    data_fp = build_data_fingerprint(train_df[cols].copy())
    cfg_fp = build_config_fingerprint(cfg, keys=["data", "lifetime_prior"])
    script_fp = build_script_fingerprint([__file__])
    cache_hit = policy == "auto" and is_cache_valid(
        paths["mixture_manifest"],
        expected_dataset=dataset,
        artifact_paths=[paths["mixture_prior"]],
        data_fingerprint=data_fp,
        config_fingerprint=cfg_fp,
        script_fingerprint=script_fp,
    )
    if cache_hit:
        print(f"[pipeline] Mixture prior cache hit: {paths['mixture_prior']}")
        prior = load_json_artifact(paths["mixture_prior"])
        return {"mixture_prior": prior if prior.get("enabled", True) else None, "raw_model": None, "cache_hit": True}
    if policy == "readonly":
        raise FileNotFoundError("lifetime_prior.cache_policy=readonly but no valid mixture cache was found.")
    if policy == "force":
        print("[pipeline] Mixture prior force rebuild requested.")
    elif policy == "off":
        print("[pipeline] Mixture prior cache disabled; fitting prior.")
    else:
        print("[pipeline] No valid mixture prior cache; fitting prior.")
    update_run_state(paths["run_state"], stage="mixture_prior", completed={"mixture_prior": False})
    result = fit_lifetime_mixture_prior(train_df, max_k=int(prior_cfg.get("max_k", 4)), dataset=dataset)
    prior = result["mixture_prior"]
    raw_model = result["raw_model"]
    save_json_artifact(prior, paths["mixture_prior"])
    save_pickle_artifact(raw_model, paths["mixture_prior_pkl"])
    manifest = build_manifest(
        dataset=dataset,
        artifact_type="mixture_prior",
        data_fingerprint=data_fp,
        config_fingerprint=cfg_fp,
        script_fingerprint=script_fp,
        artifacts={"mixture_prior": str(paths["mixture_prior"]), "mixture_pkl": str(paths["mixture_prior_pkl"])},
    )
    save_json_artifact(manifest, paths["mixture_manifest"])
    update_run_state(paths["run_state"], stage="mixture_prior", completed={"mixture_prior": True})
    return {"mixture_prior": prior if prior.get("enabled", True) else None, "raw_model": raw_model, "cache_hit": False}
