import streamlit as st
import jieba
import plotly.express as px
import pandas as pd
import re

st.set_page_config(
    page_title="AI辅助文本审核与信息质量评价系统",
    page_icon="📄",
    layout="wide"
)

# ---------------------- 加载敏感词 ----------------------
import os
def load_sensitive_words():
    words = []
    # 拿到app.py所在文件夹
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "sensitive.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    words.append(w)
    except FileNotFoundError:
        st.warning("⚠️ 未找到sensitive.txt，敏感审核功能关闭")
    return words

sensitive_list = load_sensitive_words()


def check_sensitive(text):
    for word in sensitive_list:
        if word in text:
            return True
    return False

# ---------------------- 打分函数【全面优化版】 ----------------------
def score_text(text):
    raw_text = text.strip()
    words = jieba.lcut(raw_text)
    word_count = len([w for w in words if len(w.strip())>0])
    score_completeness = 0
    score_accuracy = 25
    score_timeliness = 25
    score_valid = 0

    # ========== 1. 完整性阈值 ==========
    if word_count >= 100:
        score_completeness = 25
    elif word_count >= 60:
        score_completeness = 18
    elif word_count >= 30:
        score_completeness = 10
    elif word_count >= 8:
        score_completeness = 4
    else:
        score_completeness = 0

    # ========== 2. 基础标记 ==========
    if not raw_text:
        pure_symbol = False
    else:
        pure_symbol = all( not (c.isalpha() or c.isdigit()) for c in raw_text )

    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in raw_text)
    pure_chars = re.sub(r'[^\u4e00-\u9fff]', '', raw_text)
    words_seg = jieba.lcut(pure_chars)
    valid_word = [w for w in words_seg if len(w)>=2]
    is_nonsense = False

    # ========== 3. 噪声判定规则集 ==========
    if len(pure_chars) >= 4 and len(words_seg)>0:
        ratio_valid = len(valid_word)/len(words_seg)
        stop_words = {"的","了","是","在","和","与","也","都","就","又","有","来","过","很","比较","非常"}
        stop_cnt = sum(1 for w in words_seg if w in stop_words)
        stop_ratio = stop_cnt / len(words_seg)

        cond1 = ratio_valid < 0.3
        cond2 = (stop_ratio >= 0.4) and (ratio_valid < 0.5)
        repeat_char = re.search(r'(.)\1{3,}', pure_chars)
        cond3 = True if repeat_char else False
        punc = re.findall(r'[，。！？；]', raw_text)
        cond4 = (len(pure_chars) > 30) and (len(punc) == 0)

        domain_nouns = [
            {"飞机","航母","火箭","引擎"},
            {"开发商","楼盘","房价","物业"},
            {"奶茶","蛋糕","零食","饮料"},
            {"电脑","显卡","服务器","代码"},
            {"医院","药品","病人","医生"}
        ]
        hit_domain = set()
        for idx,group in enumerate(domain_nouns):
            for w in words_seg:
                if w in group:
                    hit_domain.add(idx)
        cond5 = len(hit_domain) >= 2

        verb_set = {"是","有","开发","购买","飞行","建设","发布","看到","进行","开展","生产","销售",
                    "觉得","感觉","认为","使用","用起来","评价","推荐","喜欢","感受","体验"}
        has_verb = any(w in verb_set for w in words_seg)
        cond6 = False
        if len(pure_chars) > 20:
            cond6 = (not has_verb) and len(valid_word)>=3

        if cond1 or cond2 or cond3 or cond4 or cond5 or cond6:
            is_nonsense = True

    # ========== 噪声分支：只对真正垃圾文本归零【重点改动】 ==========
    # 移除 word_count < 5 的强制归零！短问句、短评价允许正常打分
    if pure_symbol or (not has_chinese) or is_nonsense:
        score_valid = 0
        score_accuracy = 0
        score_timeliness = 0
        score_completeness = 0
    else:
        # ========== 正常文本分支 ==========
        if word_count >= 60:
            score_valid = 25
        elif word_count >= 30:
            score_valid = 15
        elif word_count >= 8:
            score_valid = 6
        else:
            score_valid = 0

        old_key = ["古时候","很久以前","老说法","早年","以前","前些年","过去","旧规定","老版本"]
        for k in old_key:
            if k in raw_text:
                score_timeliness = 8
                break

        uncertain_word = ["网传","据说","听说","传言","有人说","网上说","疑似","可能是","大概","好像","据爆料","小道消息"]
        for w in uncertain_word:
            if w in raw_text:
                score_accuracy = 12
                break

        sentences = re.split(r'[。！？]', raw_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 0:
            unique_sen = set(sentences)
            repeat_ratio = 1 - len(unique_sen)/len(sentences)
            if repeat_ratio >= 0.4:
                score_valid -= 8
                score_valid = max(score_valid, 0)

        empty_phrase = ["也就是说","换言之","由此可见","不难看出","综上所述","由此可得","换句话讲"]
        empty_cnt = 0
        for p in empty_phrase:
            empty_cnt += raw_text.count(p)
        score_completeness -= empty_cnt * 3
        score_completeness = max(score_completeness, 0)

    total = score_completeness + score_accuracy + score_timeliness + score_valid
    return {
        "完整性":score_completeness,
        "准确性":score_accuracy,
        "时效性":score_timeliness,
        "有效度":score_valid,
        "总分":total
    }


# ---------------------- 等级函数（原版不变） ----------------------
def get_level(score):
    if score >=80:
        return "优质信息资源"
    elif score >=60:
        return "合格信息资源"
    elif score >=40:
        return "低质信息资源"
    else:
        return "劣质无效信息"

# ---------------------- 页面UI（完全原样，无修改） ----------------------
st.title("📄 AI辅助文本审核与信息质量评价系统")
st.markdown("""
> 使用说明：复制微信公众号文章、网络评论、资讯文本粘贴到下方输入框，点击【开始评测】
> 系统自动执行：敏感合规审核 + 四维信息质量打分 + 可视化报告
""")

txt_input = st.text_area("📝 粘贴待评测文本", height=320, placeholder="在这里粘贴公众号全文、评论、新闻文本……")
run_btn = st.button("🚀 开始评测", type="primary")

if run_btn:
    if not txt_input.strip():
        st.error("请粘贴文本内容！")
    else:
        with st.spinner("正在审核、分词、计算质量分数……"):
            violate = check_sensitive(txt_input)
            if violate:
                st.error("❌ 审核结果：检测到违规内容，不予进行质量评价")
                st.metric(label="总分", value="--")
            else:
                res = score_text(txt_input)
                level = get_level(res["总分"])
                st.success("✅ 审核结果：文本合规，完成质量评价")
                col1, col2 = st.columns([1,2])
                with col1:
                    st.metric("综合总分", f"{res['总分']} / 100")
                    st.metric("质量等级", level)
                with col2:
                    radar_df = pd.DataFrame({
                        "维度":["完整性","准确性","时效性","有效度"],
                        "分数":[res["完整性"], res["准确性"], res["时效性"], res["有效度"]]
                    })
                    fig = px.line_polar(radar_df, r="分数", theta="维度", line_close=True, range_r=[0,25])
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("📊 分项得分明细")
                detail_df = pd.DataFrame([
                    {"评价维度":"完整性","分值":res["完整性"],"满分":25,"说明":"信息要素是否完备"},
                    {"评价维度":"准确性","分值":res["准确性"],"满分":25,"说明":"表述是否确定、无模糊传言"},
                    {"评价维度":"时效性","分值":res["时效性"],"满分":25,"说明":"内容是否过时陈旧"},
                    {"评价维度":"有效度","分值":res["有效度"],"满分":25,"说明":"排除灌水、无意义符号文本"},
                ])
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

                st.download_button(
                    label="📥 导出本次评测结果（csv）",
                    data=detail_df.to_csv(index=False, encoding="utf-8-sig"),
                    file_name="评测结果.csv",
                    mime="text/csv"
                )

st.divider()
st.caption("信息资源管理课程项目｜指标：完整性、准确性、时效性、有效度")
