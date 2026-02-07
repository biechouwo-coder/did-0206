"""
为总数据集添加DID变量：Treat、Post、DID

根据低碳城市试点名单设定三个变量：
1. Treat (分组变量)：城市在任何一批名单中出现过为1，否则为0
2. Post (时间变量)：对于试点城市，年份≥政策实施年份为1，否则为0
3. DID (政策变量)：Treat × Post
"""
import pandas as pd
from pathlib import Path

# 定义低碳城市试点名单及其政策实施年份
PILOT_CITIES = {
    # 第一批 (2009年政策实施，但根据用户描述是2010年，这里按用户要求使用2010)
    "天津市": 2010, "重庆市": 2010, "厦门市": 2010, "杭州市": 2010,
    "南昌市": 2010, "贵阳市": 2010, "保定市": 2010,
    # 广东省
    "深圳市": 2010, "广州市": 2010, "珠海市": 2010, "汕头市": 2010,
    "佛山市": 2010, "江门市": 2010, "湛江市": 2010, "茂名市": 2010,
    "肇庆市": 2010, "惠州市": 2010, "梅州市": 2010, "汕尾市": 2010,
    "河源市": 2010, "阳江市": 2010, "清远市": 2010, "东莞市": 2010,
    "中山市": 2010, "潮州市": 2010, "揭阳市": 2010, "云浮市": 2010,
    "韶关市": 2010,
    # 辽宁省
    "沈阳市": 2010, "大连市": 2010, "鞍山市": 2010, "抚顺市": 2010,
    "本溪市": 2010, "丹东市": 2010, "锦州市": 2010, "营口市": 2010,
    "阜新市": 2010, "辽阳市": 2010, "盘锦市": 2010, "铁岭市": 2010,
    "朝阳市": 2010, "葫芦岛市": 2010,
    # 湖北省
    "武汉市": 2010, "黄石市": 2010, "十堰市": 2010, "宜昌市": 2010,
    "襄樊市": 2010, "鄂州市": 2010, "荆门市": 2010, "孝感市": 2010,
    "荆州市": 2010, "黄冈市": 2010, "咸宁市": 2010, "随州市": 2010,
    "恩施土家族苗族自治州": 2010,
    # 陕西省
    "西安市": 2010, "铜川市": 2010, "宝鸡市": 2010, "咸阳市": 2010,
    "渭南市": 2010, "延安市": 2010, "汉中市": 2010, "榆林市": 2010,
    "安康市": 2010, "商洛市": 2010,
    # 云南省
    "昆明市": 2010, "曲靖市": 2010, "玉溪市": 2010, "保山市": 2010,
    "昭通市": 2010, "丽江市": 2010, "普洱市": 2010, "临沧市": 2010,
    "楚雄彝族自治州": 2010, "红河哈尼族彝族自治州": 2010,
    "文山壮族苗族自治州": 2010, "西双版纳傣族自治州": 2010,
    "大理白族自治州": 2010, "德宏傣族景颇族自治州": 2010,
    "怒江傈僳族自治州": 2010, "迪庆藏族自治州": 2010,

    # 第二批 (2012年)
    "海口市": 2012, "三亚市": 2012,
    "北京市": 2012, "上海市": 2012, "石家庄市": 2012, "秦皇岛市": 2012,
    "晋城市": 2012, "呼伦贝尔市": 2012, "吉林市": 2012, "大兴安岭地区": 2012,
    "苏州市": 2012, "淮安市": 2012, "镇江市": 2012, "宁波市": 2012,
    "温州市": 2012, "青岛市": 2012, "济南市": 2012, "金昌市": 2012,
    "乌鲁木齐市": 2012, "桂林市": 2012,

    # 第三批 (2017年)
    "乌海市": 2017, "南京市": 2017, "常州市": 2017, "无锡市": 2017,
    "嘉兴市": 2017, "合肥市": 2017, "淮北市": 2017, "福州市": 2017,
    "南平市": 2017, "吉安市": 2017, "萍乡市": 2017, "淄博市": 2017,
    "郑州市": 2017, "焦作市": 2017, "三门峡市": 2017, "长沙市": 2017,
    "株洲市": 2017, "湘潭市": 2017, "郴州市": 2017, "柳州市": 2017,
    "成都市": 2017, "广元市": 2017, "拉萨市": 2017, "兰州市": 2017,
    "西宁市": 2017, "银川市": 2017, "吴忠市": 2017,
    "昌吉回族自治州": 2017, "和田地区": 2017, "喀什地区": 2017,
}


def add_did_variables(df, city_col='city_name', year_col='year'):
    """
    为数据集添加DID变量

    Parameters:
    -----------
    df : pd.DataFrame
        原始数据集
    city_col : str
        城市列名，默认为'city_name'
    year_col : str
        年份列名，默认为'year'

    Returns:
    --------
    pd.DataFrame
        添加了Treat、Post、DID变量的数据集
    """
    df = df.copy()

    # 1. Treat变量：如果城市在任何一批名单中出现过则为1，否则为0
    df['Treat'] = df[city_col].map(lambda x: 1 if x in PILOT_CITIES else 0)

    # 2. Post变量：对于试点城市，如果年份≥政策实施年份则为1，否则为0
    # 非试点城市始终为0
    def get_post_year(row):
        city = row[city_col]
        if city in PILOT_CITIES:
            policy_year = PILOT_CITIES[city]
            return 1 if row[year_col] >= policy_year else 0
        return 0

    df['Post'] = df.apply(get_post_year, axis=1)

    # 3. DID变量：Treat × Post
    df['DID'] = df['Treat'] * df['Post']

    return df


def main():
    # 文件路径
    base_dir = Path(r"c:\Users\HP\Desktop\did-0206")
    input_file = base_dir / "总数据集_已合并_含碳排放_new.xlsx"
    output_file = base_dir / "总数据集_已合并_含碳排放_new.xlsx"

    print("=" * 70)
    print("为总数据集添加DID变量")
    print("=" * 70)

    # 读取数据
    print(f"\n正在读取数据文件：{input_file}")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"读取失败：{e}")
        return

    print(f"数据形状：{df.shape}")
    print(f"列名：{df.columns.tolist()}")

    # 检查必要的列是否存在
    city_col = None
    year_col = None

    # 优先使用英文列名
    if 'city_name' in df.columns:
        city_col = 'city_name'
    elif '城市' in df.columns:
        city_col = '城市'
    else:
        # 查找包含'city'或'城市'的列
        for col in df.columns:
            if 'city' in str(col).lower() or '城市' in str(col):
                city_col = col
                break

    if 'year' in df.columns:
        year_col = 'year'
    elif '年份' in df.columns:
        year_col = '年份'
    else:
        # 查找包含'year'或'年份'的列
        for col in df.columns:
            if 'year' in str(col).lower() or '年份' in str(col):
                year_col = col
                break

    if city_col is None:
        print("\n错误：未找到城市列！")
        print(f"可用列：{df.columns.tolist()}")
        return

    if year_col is None:
        print("\n错误：未找到年份列！")
        print(f"可用列：{df.columns.tolist()}")
        return

    print(f"\n识别到城市列：{city_col}")
    print(f"识别到年份列：{year_col}")

    # 添加DID变量
    print("\n正在添加DID变量...")
    df = add_did_variables(df, city_col=city_col, year_col=year_col)

    # 显示统计信息
    print("\n" + "=" * 70)
    print("DID变量统计")
    print("=" * 70)

    print(f"\nTreat变量分布：")
    print(df['Treat'].value_counts().to_string())

    print(f"\nPost变量分布：")
    print(df['Post'].value_counts().to_string())

    print(f"\nDID变量分布：")
    print(df['DID'].value_counts().to_string())

    # 试点城市数量
    pilot_cities_in_data = df[df[city_col].isin(PILOT_CITIES.keys())][city_col].nunique()
    print(f"\n数据集中的试点城市数量：{pilot_cities_in_data}")
    print(f"试点名单中的城市数量：{len(PILOT_CITIES)}")

    # 展示部分数据
    print("\n" + "=" * 70)
    print("示例数据（前10行）")
    print("=" * 70)
    display_cols = [city_col, year_col, 'Treat', 'Post', 'DID']
    print(df[display_cols].head(10).to_string(index=False))

    # 保存数据
    print(f"\n正在保存到：{output_file}")
    df.to_excel(output_file, index=False, engine='openpyxl')

    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)

    return df


if __name__ == "__main__":
    df = main()
