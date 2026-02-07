"""
Generate Excel file listing all low-carbon pilot cities in China.
"""
import pandas as pd
from collections import Counter

def generate_low_carbon_cities():
    # Define the data
    first_batch_cities = [
        # 7单独试点
        "天津市", "重庆市", "厦门市", "杭州市", "南昌市", "贵阳市", "保定市",
        # 21广东省
        "深圳市", "广州市", "珠海市", "汕头市", "佛山市", "江门市", "湛江市", "茂名市",
        "肇庆市", "惠州市", "梅州市", "汕尾市", "河源市", "阳江市", "清远市",
        "东莞市", "中山市", "潮州市", "揭阳市", "云浮市", "韶关市",
        # 14辽宁省
        "沈阳市", "大连市", "鞍山市", "抚顺市", "本溪市", "丹东市", "锦州市", "营口市",
        "阜新市", "辽阳市", "盘锦市", "铁岭市", "朝阳市", "葫芦岛市",
        # 13湖北省
        "武汉市", "黄石市", "十堰市", "宜昌市", "襄樊市", "鄂州市", "荆门市", "孝感市",
        "荆州市", "黄冈市", "咸宁市", "随州市", "恩施土家族苗族自治州",
        # 10陕西省
        "西安市", "铜川市", "宝鸡市", "咸阳市", "渭南市", "延安市", "汉中市",
        "榆林市", "安康市", "商洛市",
        # 16云南省
        "昆明市", "曲靖市", "玉溪市", "保山市", "昭通市", "丽江市", "普洱市", "临沧市",
        "楚雄彝族自治州", "红河哈尼族彝族自治州", "文山壮族苗族自治州",
        "西双版纳傣族自治州", "大理白族自治州", "德宏傣族景颇族自治州",
        "怒江傈僳族自治州", "迪庆藏族自治州"
    ]

    second_batch_cities = [
        # 2海南省
        "海口市", "三亚市",
        # 18单独试点
        "北京市", "上海市", "石家庄市", "秦皇岛市", "晋城市", "呼伦贝尔市",
        "吉林市", "大兴安岭地区", "苏州市", "淮安市", "镇江市", "宁波市",
        "温州市", "青岛市", "济南市", "金昌市", "乌鲁木齐市", "桂林市"
    ]

    third_batch_cities = [
        "乌海市", "南京市", "常州市", "无锡市", "嘉兴市", "合肥市", "淮北市",
        "福州市", "南平市", "吉安市", "萍乡市", "淄博市", "郑州市", "焦作市",
        "三门峡市", "长沙市", "株洲市", "湘潭市", "郴州市", "柳州市", "成都市",
        "广元市", "拉萨市", "兰州市", "西宁市", "银川市", "吴忠市",
        "昌吉回族自治州", "和田地区", "喀什地区"
    ]

    # Create data structure
    data = []

    # Add first batch cities
    for city in first_batch_cities:
        data.append({
            "城市名": city,
            "政策实施年": 2009,
            "批次": "第一批"
        })

    # Add second batch cities
    for city in second_batch_cities:
        data.append({
            "城市名": city,
            "政策实施年": 2012,
            "批次": "第二批"
        })

    # Add third batch cities
    for city in third_batch_cities:
        data.append({
            "城市名": city,
            "政策实施年": 2017,
            "批次": "第三批"
        })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by batch (custom order) and then by city name
    batch_order = {"第一批": 1, "第二批": 2, "第三批": 3}
    df["批次排序"] = df["批次"].map(batch_order)
    df = df.sort_values(["批次排序", "城市名"])
    df = df.drop("批次排序", axis=1)

    # Check for duplicates
    city_counts = Counter(df["城市名"])
    duplicates = {city: count for city, count in city_counts.items() if count > 1}

    # Print statistics
    print("=" * 60)
    print("低碳城市试点统计")
    print("=" * 60)
    print(f"\n总城市数: {len(df)}")
    print(f"第一批 (2009年): {len(first_batch_cities)} 个城市")
    print(f"第二批 (2012年): {len(second_batch_cities)} 个城市")
    print(f"第三批 (2017年): {len(third_batch_cities)} 个城市")
    print(f"预期总计: {len(first_batch_cities) + len(second_batch_cities) + len(third_batch_cities)} 个城市")

    if duplicates:
        print(f"\n发现重复城市: {len(duplicates)} 个")
        for city, count in duplicates.items():
            print(f"  - {city}: {count} 次")
    else:
        print("\n未发现重复城市")

    print("\n" + "=" * 60)
    print("各批次城市明细")
    print("=" * 60)
    for batch in ["第一批", "第二批", "第三批"]:
        batch_df = df[df["批次"] == batch]
        year = batch_df["政策实施年"].iloc[0]
        print(f"\n{batch} ({year}年): {len(batch_df)} 个城市")
        print(f"  前5个城市: {', '.join(batch_df['城市名'].head(5).tolist())}...")

    # Save to Excel
    output_path = r"c:\Users\HP\Desktop\did-0206\低碳城市试点名单.xlsx"
    df.to_excel(output_path, index=False, engine='openpyxl')

    print(f"\n" + "=" * 60)
    print(f"Excel文件已保存至: {output_path}")
    print("=" * 60)

    return df, duplicates

if __name__ == "__main__":
    df, duplicates = generate_low_carbon_cities()
