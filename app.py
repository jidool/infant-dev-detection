# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import anthropic

# ── 페이지 설정 ───────────────────────────────────────────
st.set_page_config(
    page_title="영유아 발달 지연 탐지 시스템",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 색상 팔레트 ───────────────────────────────────────────
COLOR = {
    '정상_발달군': '#4BBFA5',
    '경계선군':   '#F5A623',
    '지연_의심군': '#F47B7B',
    'A형': '#4BBFA5', 'B형': '#F5A623',
    'C형': '#7B9EF4', 'D형': '#F47B7B',
    'bg': '#FAFAF7'
}

# ── 데이터 로드 ───────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/환경변수_통합_분석결과.csv', encoding='utf-8-sig')
    except FileNotFoundError:
        # 배포 환경: 샘플 데이터 생성
        np.random.seed(42)
        n = 973
        df = pd.DataFrame({
            'N_ID': range(1, n+1),
            'score_fm': np.random.choice([0,1,2], n, p=[0.05,0.2,0.75]),
            'score_ps': np.random.choice([0,1,2], n, p=[0.05,0.2,0.75]),
            'score_cog': np.random.normal(107, 12, n).clip(60,130),
            'score_lng': np.random.normal(109, 12, n).clip(60,130),
            'cluster_label': np.random.choice(
                ['정상_발달군','경계선군','지연_의심군'], n, p=[0.544,0.243,0.213]),
            'dev_type': np.random.choice(['A형','B형','C형','D형'], n, p=[0.288,0.227,0.227,0.258]),
            'risk_score': np.random.exponential(15, n).clip(0,100),
            'risk_grade': np.random.choice(['정상','경계','위험'], n, p=[0.892,0.101,0.007]),
            'motor_social': np.random.normal(1.86, 0.2, n).clip(0,2),
            'cog_lang': np.random.normal(107, 12, n).clip(60,130),
            'target': np.random.choice([0,1,2], n, p=[0.213,0.243,0.544]),
            '성별': np.random.choice([1,2], n),
            'income': np.random.randint(1,12,n),
            'mom_edu': np.random.randint(1,9,n),
            'care_type': np.random.choice([0,1,2,3,4,5], n),
            'main_care': np.random.choice([1,2,3], n),
        })
    df['성별'] = pd.to_numeric(df['성별'], errors='coerce')
    df['성별_라벨'] = df['성별'].map({1: '남', 2: '여'}).fillna('미상')
    return df

df = load_data()

# ── 모델 학습 (교사 입력 예측용) ─────────────────────────
@st.cache_resource
def train_model():
    FEATURES = ['score_fm', 'score_ps', 'score_cog', 'score_lng']
    X = df[FEATURES].values
    # cluster 컬럼 없으면 cluster_label로 대체 생성
    if 'cluster' not in df.columns:
        label_map = {'지연_의심군': 0, '경계선군': 1, '정상_발달군': 2}
        y = df['cluster_label'].map(label_map).values
    else:
        y = df['cluster'].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=300, random_state=42,
                                class_weight='balanced')
    rf.fit(X_sc, y)
    return rf, scaler

rf_model, scaler = train_model()

# ── 사이드바 ──────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/children.png", width=60)
st.sidebar.title("🌱 발달 지연 탐지")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "메뉴",
    ["📊 전체 대시보드", "🔍 아동 개별 분석", "👩‍🏫 교사 입력 도구"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.caption("숙명여자대학교 빅데이터분석\n한국아동패널 3·5차(2010·2012)\nn=973명")

# ══════════════════════════════════════════════════════════
# PAGE 1: 전체 대시보드
# ══════════════════════════════════════════════════════════
if page == "📊 전체 대시보드":

    st.title("📊 영유아 발달 지연 위험군 탐지 대시보드")
    st.caption("한국아동패널 3·5차 데이터 기반 | 발달 군집 분류 + 위험도 스코어")
    st.markdown("---")

    # ── KPI 카드 ─────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("전체 아동", f"{len(df):,}명")
    k2.metric("지연 의심군", f"{(df['cluster_label']=='지연_의심군').sum()}명",
              f"{(df['cluster_label']=='지연_의심군').mean()*100:.1f}%")
    k3.metric("경계선군", f"{(df['cluster_label']=='경계선군').sum()}명",
              f"{(df['cluster_label']=='경계선군').mean()*100:.1f}%")
    k4.metric("평균 위험도", f"{df['risk_score'].mean():.1f}점",
              f"최대 {df['risk_score'].max():.0f}점")

    st.markdown("---")

    # ── Row 1: 군집 분포 + 발달유형 분포 ─────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("군집별 분포")
        cluster_cnt = df['cluster_label'].value_counts().reindex(
            ['정상_발달군','경계선군','지연_의심군'])
        fig = px.bar(
            x=cluster_cnt.index, y=cluster_cnt.values,
            color=cluster_cnt.index,
            color_discrete_map=COLOR,
            text=[f"{v}명\n({v/len(df)*100:.1f}%)" for v in cluster_cnt.values],
            labels={'x':'군집','y':'아동 수'}
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, plot_bgcolor=COLOR['bg'],
                          paper_bgcolor=COLOR['bg'], height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("발달 유형별 분포 (A/B/C/D형)")
        type_cnt = df['dev_type'].value_counts().sort_index()
        fig2 = px.pie(
            values=type_cnt.values, names=type_cnt.index,
            color=type_cnt.index,
            color_discrete_map=COLOR,
            hole=0.4
        )
        fig2.update_traces(textinfo='label+percent')
        fig2.update_layout(plot_bgcolor=COLOR['bg'],
                           paper_bgcolor=COLOR['bg'], height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: 위험도 히스토그램 + 성별 비교 ─────────────
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("위험도 스코어 분포")
        fig3 = px.histogram(
            df, x='risk_score', color='cluster_label',
            color_discrete_map=COLOR, nbins=40,
            labels={'risk_score':'위험도 스코어','count':'아동 수'},
            barmode='overlay', opacity=0.75
        )
        fig3.update_layout(plot_bgcolor=COLOR['bg'],
                           paper_bgcolor=COLOR['bg'], height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.subheader("성별 × 군집 비교")
        gender_cross = df.groupby(['성별_라벨','cluster_label']).size().reset_index(name='count')
        fig4 = px.bar(
            gender_cross, x='성별_라벨', y='count',
            color='cluster_label', color_discrete_map=COLOR,
            barmode='group',
            labels={'성별_라벨':'성별','count':'아동 수','cluster_label':'군집'}
        )
        fig4.update_layout(plot_bgcolor=COLOR['bg'],
                           paper_bgcolor=COLOR['bg'], height=350)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: 영역별 점수 비교 레이더 ───────────────────
    st.subheader("군집별 발달 영역 점수 비교")
    radar_data = df.groupby('cluster_label')[
        ['score_fm','score_ps','score_cog','score_lng']].mean()
    categories = ['소근육','사회성','인지','언어']

    fig5 = go.Figure()
    for cl, color in COLOR.items():
        if cl not in ['정상_발달군','경계선군','지연_의심군']:
            continue
        row = radar_data.loc[cl].values.tolist()
        # 정규화 (0~1)
        row_norm = [(v - df[c].min()) / (df[c].max() - df[c].min())
                    for v, c in zip(row, ['score_fm','score_ps','score_cog','score_lng'])]
        fig5.add_trace(go.Scatterpolar(
            r=row_norm + [row_norm[0]],
            theta=categories + [categories[0]],
            fill='toself', name=cl,
            line_color=color, opacity=0.7
        ))
    fig5.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        paper_bgcolor=COLOR['bg'], height=400
    )
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 2: 아동 개별 분석
# ══════════════════════════════════════════════════════════
elif page == "🔍 아동 개별 분석":

    st.title("🔍 아동 개별 위험도 조회")
    st.markdown("---")

    # ── 필터 ─────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        sel_cluster = st.multiselect(
            "군집 필터",
            options=['정상_발달군','경계선군','지연_의심군'],
            default=['경계선군','지연_의심군']
        )
    with f2:
        sel_type = st.multiselect(
            "발달유형 필터",
            options=['A형','B형','C형','D형'],
            default=['C형','D형']
        )
    with f3:
        risk_range = st.slider("위험도 스코어 범위", 0, 100, (20, 100))

    # ── 필터 적용 ─────────────────────────────────────────
    mask = (
        df['cluster_label'].isin(sel_cluster) &
        df['dev_type'].isin(sel_type) &
        df['risk_score'].between(*risk_range)
    )
    df_filtered = df[mask].sort_values('risk_score', ascending=False)

    st.markdown(f"**{len(df_filtered)}명 해당**")

    # ── 테이블 ────────────────────────────────────────────
    display_cols = {
        'N_ID': 'ID', 'cluster_label': '군집', 'dev_type': '발달유형',
        'risk_score': '위험도', 'risk_grade': '등급',
        'score_cog': '인지', 'score_lng': '언어',
        'score_fm': '소근육', 'score_ps': '사회성', '성별_라벨': '성별'
    }
    st.dataframe(
        df_filtered[list(display_cols.keys())].rename(columns=display_cols),
        use_container_width=True, height=450,
        column_config={
            '위험도': st.column_config.ProgressColumn(
                '위험도', min_value=0, max_value=100, format='%.1f')
        }
    )


# ══════════════════════════════════════════════════════════
# PAGE 3: 교사 입력 도구
# ══════════════════════════════════════════════════════════
elif page == "👩‍🏫 교사 입력 도구":

    st.title("👩‍🏫 교사용 발달 진단 도구")
    st.caption("아동의 발달 점수를 입력하면 군집·위험도·교육적 개입 방향을 즉시 제공합니다.")
    st.markdown("---")

    col_input, col_result = st.columns([1, 1.2])

    with col_input:
        st.subheader("📝 아동 정보 입력")
        child_name = st.text_input("아동 이름 (선택)", placeholder="예: 홍길동")
        gender = st.selectbox("성별", ["남", "여"])

        st.markdown("**Denver II 발달 점수**")
        score_fm = st.slider("소근육 점수", 0.0, 2.0, 1.8, 0.1,
                             help="0=지연, 1=주의, 2=정상")
        score_ps = st.slider("사회성 점수", 0.0, 2.0, 1.8, 0.1,
                             help="0=지연, 1=주의, 2=정상")

        st.markdown("**K-DST 표준 점수**")
        score_cog = st.slider("인지 표준점수", 60, 130, 100, 1)
        score_lng = st.slider("언어 표준점수", 60, 130, 100, 1)

        api_key = st.text_input("Claude API Key (개입 방향 생성용)",
                                type="password",
                                placeholder="sk-ant-...")
        run = st.button("🔍 분석 실행", type="primary", use_container_width=True)

    with col_result:
        if run:
            # ── 군집 예측 ─────────────────────────────────
            X_input = scaler.transform([[score_fm, score_ps, score_cog, score_lng]])
            pred_cluster = rf_model.predict(X_input)[0]
            pred_proba   = rf_model.predict_proba(X_input)[0]

            cluster_map = {0: '지연_의심군', 1: '경계선군', 2: '정상_발달군'}
            cluster_label = cluster_map[pred_cluster]

            # ── 위험도 스코어 ─────────────────────────────
            w = {'lng': 0.30, 'cog': 0.25, 'ps': 0.25, 'fm': 0.20}
            lng_n = max(0, (130 - score_lng) / 70)
            cog_n = max(0, (130 - score_cog) / 70)
            ps_n  = max(0, (2 - score_ps) / 2)
            fm_n  = max(0, (2 - score_fm) / 2)
            risk  = (w['lng']*lng_n + w['cog']*cog_n +
                     w['ps']*ps_n   + w['fm']*fm_n) * 100

            # ── 발달유형 ──────────────────────────────────
            med_motor = df['motor_social'].median()
            med_cog   = df['cog_lang'].median()
            motor = (score_fm + score_ps) / 2
            cog   = (score_cog + score_lng) / 2
            if   motor >= med_motor and cog >= med_cog:   dev_type = 'A형 (균형 정상)'
            elif motor <  med_motor and cog >= med_cog:   dev_type = 'B형 (운동·사회성 지연)'
            elif motor >= med_motor and cog <  med_cog:   dev_type = 'C형 (인지·언어 지연)'
            else:                                          dev_type = 'D형 (전반 지연)'

            # ── 결과 표시 ─────────────────────────────────
            st.subheader("📋 분석 결과")

            r1, r2 = st.columns(2)
            c_color = COLOR[cluster_label]
            r1.markdown(f"""
            <div style='background:{c_color}22; border-left:4px solid {c_color};
                        padding:12px; border-radius:8px;'>
                <b>군집</b><br>
                <span style='font-size:1.4em; font-weight:bold; color:{c_color};'>
                {cluster_label}</span>
            </div>""", unsafe_allow_html=True)
            r2.markdown(f"""
            <div style='background:#F0F0F0; border-left:4px solid #888;
                        padding:12px; border-radius:8px;'>
                <b>발달 유형</b><br>
                <span style='font-size:1.2em; font-weight:bold;'>
                {dev_type}</span>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # 위험도 게이지
            st.markdown(f"**위험도 스코어: {risk:.1f} / 100**")
            gauge_color = '#4BBFA5' if risk < 20 else '#F5A623' if risk < 40 else '#F47B7B'
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk,
                gauge=dict(
                    axis=dict(range=[0,100]),
                    bar=dict(color=gauge_color),
                    steps=[
                        dict(range=[0,20],  color='#E8F8F5'),
                        dict(range=[20,40], color='#FEF9E7'),
                        dict(range=[40,100],color='#FDEDEC'),
                    ]
                )
            ))
            fig_g.update_layout(height=220, margin=dict(t=20,b=10,l=20,r=20),
                                paper_bgcolor=COLOR['bg'])
            st.plotly_chart(fig_g, use_container_width=True)

            # 예측 확률
            st.markdown("**군집 예측 확률**")
            for cl, prob in zip(['지연_의심군','경계선군','정상_발달군'], pred_proba):
                st.progress(float(prob), text=f"{cl}: {prob*100:.1f}%")

            # ── Claude API 개입 방향 ───────────────────────
            st.markdown("---")
            st.subheader("💡 교육적 개입 방향")

            if api_key:
                with st.spinner("Claude가 개입 방향을 생성하는 중..."):
                    try:
                        client = anthropic.Anthropic(api_key=api_key)
                        prompt = f"""
당신은 영유아 발달 전문가입니다. 아래 아동의 발달 데이터를 바탕으로
어린이집·유치원 교사가 실천할 수 있는 교육적 개입 방향을 제시해주세요.

[아동 정보]
- 이름: {child_name if child_name else '미입력'}
- 성별: {gender}
- 군집: {cluster_label}
- 발달 유형: {dev_type}
- 위험도 스코어: {risk:.1f}/100
- 소근육: {score_fm}/2.0
- 사회성: {score_ps}/2.0
- 인지 표준점수: {score_cog}
- 언어 표준점수: {score_lng}

[이론적 배경]
Vygotsky ZPD, Bronfenbrenner 생태학적 체계이론, Bloom 결정적 시기

다음 형식으로 한국어로 답해주세요:
1. **현재 발달 상태 요약** (2~3문장)
2. **즉각적 개입 전략** (3가지, 교실에서 바로 실천 가능한 것)
3. **부모 연계 방향** (1~2가지)
4. **전문기관 연계 필요 여부** (예/아니오 + 이유)
"""
                        response = client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=1000,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.markdown(response.content[0].text)
                    except Exception as e:
                        st.error(f"API 오류: {e}")
            else:
                # Fallback 텍스트
                fallback = {
                    '정상_발달군': """
**현재 발달 상태 요약**
전반적으로 균형 잡힌 발달을 보이고 있습니다. 인지·언어·운동·사회성 모든 영역이 정상 범위에 있습니다.

**즉각적 개입 전략**
1. 자유놀이 시간을 충분히 제공하여 주도적 탐색을 장려하세요.
2. 또래 협동 활동을 통해 사회성을 더욱 강화하세요.
3. 언어 확장 활동(이야기 꾸미기, 책 읽기)을 일과에 포함하세요.

**부모 연계 방향**
현재 발달 상태를 긍정적으로 공유하고, 가정에서도 독서·대화 환경을 유지하도록 안내하세요.

**전문기관 연계 필요 여부**
아니오 — 현재는 정기 모니터링으로 충분합니다.""",
                    '경계선군': """
**현재 발달 상태 요약**
운동·사회성 영역에서 또래 대비 상대적으로 낮은 수행을 보이고 있습니다. 조기 개입 시 정상 발달군으로 회복 가능성이 높습니다.

**즉각적 개입 전략**
1. 소그룹 활동(3~4명)을 통해 또래 상호작용 기회를 늘리세요.
2. 미세 소근육 활동(가위질, 점토, 구슬꿰기)을 일과에 추가하세요.
3. ZPD를 고려한 비계 설정 — 약간 도전적인 과제를 교사 지원과 함께 제공하세요.

**부모 연계 방향**
가정에서 함께하는 신체놀이(공 던지기, 블록쌓기)를 권장하고, 한 달 후 재평가 일정을 안내하세요.

**전문기관 연계 필요 여부**
조건부 — 2개월 후 재평가 시 개선이 없으면 발달센터 연계를 권장합니다.""",
                    '지연_의심군': """
**현재 발달 상태 요약**
인지·언어 영역에서 또래 대비 유의미한 지연이 관찰됩니다. Bloom의 결정적 시기를 고려할 때 즉각적인 개입이 필요합니다.

**즉각적 개입 전략**
1. 1:1 언어 집중 활동을 하루 10~15분 실시하세요 (그림 보고 설명하기, 단어 확장).
2. 교사 주도 구조화된 활동으로 인지 자극을 규칙적으로 제공하세요.
3. 성공 경험을 자주 만들어 학습 동기와 자존감을 지지하세요.

**부모 연계 방향**
발달 상황을 솔직하고 따뜻하게 공유하고, 가정 내 언어 환경 개선(TV 줄이기, 대화 늘리기)을 구체적으로 안내하세요.

**전문기관 연계 필요 여부**
예 — 지역 육아종합지원센터 또는 발달지원센터에 조기 의뢰를 권장합니다."""
                }
                st.markdown(fallback.get(cluster_label, ''))
                st.info("💡 Claude API Key를 입력하면 아동별 맞춤 개입 방향을 생성할 수 있습니다.")
        else:
            st.info("← 왼쪽에서 아동 점수를 입력하고 '분석 실행'을 눌러주세요.")
