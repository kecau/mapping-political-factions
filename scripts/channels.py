"""The 96 Korean political YouTube channels that make up the study set.

Channel ids are public YouTube identifiers. The trailing comment on each line
is the channel's display title at the time of collection, kept for readability;
the authoritative title and political leaning (``wing``) are read from the
``youtube_channels`` table at analysis time, not from this file.

The list is ordered left-leaning channels first, then right-leaning ones, with
a blank line separating the two blocks. That ordering carries no meaning for
the analysis -- every wing assignment comes from the database.
"""

CHANNEL_IDS = [

    "UCAAvO0ehWox1bbym3rXKBZw", # 김어준의 겸손은힘들다 뉴스공장
    "UCgeOlLcX6PReHdWImEnUVTg",  # 스픽스
    "UCMYhq9OyGI5UEz_NTAoHY7A",  # [팟빵] 최욱의 매불쇼
    "UCg11mut6P5ifydCeZVhVD1Q",  # 민주는 파출부
    "UCeI9fSiNXhvslhzVO0gZVcw",  # 파란공장
    "UC8tJM0XF53xTSWXeMLCUoeg",  # 세상경사
    "UC-hc6yLEjk5C5gMTJKFGlPw",  # 정치 읽어주는 여자
    "UC-VnP-VdGUaIbssXoOhz5OQ",  # 언론 알아야 바꾼다
    "UCYUGo6ZfQZCTMrK9rMDijNQ",  # 뉴스토마토
    "UCljnbFCt-4doBr7wtEIIbbw",  # 김용민TV
    "UCjV691vj7lwcMJNGgP4K5jw",  # 정치고고
    "UCi-6JuoDiO6W9BsdTBmUA2Q",  # 민주피플
    "UCX7-K_PSdtAiUDLEMQwrRoQ",  # 고발뉴스TV
    "UCyYWbRzKBQxeXYeddPLpOvA",  # 국민이 주인이다
    "UCijojqRd_6p21wtI13fE68w",  # 민주데이
    "UCHNTac9DuW4v9tqT-S1b25w",  # 푸른정치
    "UC-KaVRIF9KXi08lKNwX4X6g",  # 시민 김상식TV
    "UCuka4cOMWar3MM0WDtE9ehw",  # 애국청년김태풍
    "UCnj0yVadEmXKcUN5ztUY4uw",  # 백운기의 정어리TV
    "UCBGZwqjq8t-XZPPj_kM9EaQ",  # 바꿀수만있다면TV
    "UCNJM6dqu70Qr6VaseiW1Org",  # 이재명
    "UCbouR-45rChK6ou95zSlNxA",  # 사이다 이슈-시사뉴스
    "UCcfnM_fIX1Q_ikHV7dbGI_w",  # 최한욱TV
    "UCKQI9j2Rh4kP1w65WnreqMA",  # 짤짤정치
    "UCIz3v90gtyLeDjIKvricbrw",  # 이야기창
    "UCu1FzjrHosuKGvgIx8oBi8w",  # [공식] 새날
    "UCN7rkrXKCddsdKqH9xBiUTw",  # 시사건건
    "UCRXo2A0wih37dni4jTVZi_g",  # 이슈읽기
    "UCL9WH7kxmAd1hhNDp2bsXig",  # 정치썰집
    "UCUFpe4kv7GRPEddLfCJOIzQ",  # 사장남천동
    "UCpr8CBjls1XYoSd98d6aT1w",  # 뉴탐사 NewTamsa
    "UCAVVxLmPDFkSTROPue8ZrRA",  # 장윤선의 취재편의점
    "UCdrYtMQv4CiyWO_UkFvw4fw",  # 정치한잔 나발독립군
    "UCIMv9bOOGWGIfg6wPcRLItQ",  # 박시영TV
    "UCzQJmmpZjqzJe96CwlrwlHQ",  # 시사타파TV
    "UCGChSHqn7VubkfsUH3ZuwbA",  # 한입정치
    "UCIJ2nADr4SvtPrVAL07x-ZA",  # 잼며든다
    "UCwFcwmKmiwj69Btk0hvZF0g",  # 김상욱TV
    "UC9mEeE55q4PCfGERt_PNYbg",  # 봉지욱의 오프더레코드
    "UCzuBleEdJox6p_p5wexcp5A",  # 정치일주
    "UCd1rmndnoWPVRSuY_K7Ferw",  # 파란불도저
    "UC0b7pdHf7Bs2DA9OVUe0XQA",  # 뉴스반장
    "UC5eYa2PEgbrV6YLiH04Ws2A",  # 타이거행크TV
    "UCDxxW9v5yxNAPHUCrBLo1jA",  # 더얌마
    "UCyeVEuBHGDRZGt4krdMOhMA",  # 헬마라이브
    "UCsuOO7Qws7gtknOW8H0q9hw",  # 60인사이트
    "UCZtpn1JjS0OX21uEeNjDEsA",  # 시사맛집
    "UCj8Snyrs1y-wnBQiUmGrTjw",  # KTV 이매진
    "UCFPEOFLdrchYyyBY5ln5AFQ",  # 지켜라민주

    "UCOqCunaF9qVN8bXwsK0HT3g",  # 펜앤마이크TV
    "UCFWIofVBqseX6Bmd9Dn7kgg",  # 성창경TV
    "UCNzWZmsJ2QmBss30LeZZTdg",  # 서정욱TV
    "UCris9QMbviXz-Zh__wA4JFA",  # 한동훈
    "UC8SZ88GQa9XWwZxLNibO3ug",  # 배승희 변호사
    "UClUlAMZGEGPn8mKemiwOjzg",  # GROUND C 그라운드씨
    "UCSd6jlFPB6iVG3eCTH7XJZA",  # 시사포커스TV
    "UCbgqDFODvh-q38ouzoKKkVA",  # 신인균의 국방TV
    "UC-U6-u6w-zNM_w6ilDVYqHQ",  # 송국건의 혼술
    "UCM8BcGB6BWKq3utIMhGKnUA",  # 고성국TV
    "UCXvQXTcC77Nav46NYc4FtDA",  # 어벤저스전략회의
    "UCBnJW5kVSSarqXkUxzRuDBg",  # 하경숙의 아침TV
    "UCQbwZn5Lios_LJRQSrVXPyQ",  # 이대남의우회전
    "UCDfZSAhIHRenIBCQt_AQ-Rg",  # 이영돈TV
    "UCuCECsh_wb-C4loI4xboWVw",  # 젊은시각
    "UCI788NxeJn_gr5GlzmnD3vw",  # 손상대TV2
    "UCxuf3GXK290vcpFW0lxm0Uw",  # 이봉규TV
    "UC-dVHhMUEaXuNb8efhis1dg",  # BJ톨
    "UCR3zvEwGCwy22mEgbe-7HIw",  # [정광용 TV]레지스탕스TV
    "UCH_Bxedipg--u3iavdM57rg",  # 샤인튜브
    "UCALJIrWxBI68-JI84Jge0ag",  # 에스더의 저녁tv
    "UCUFxsBn6wIRsfW0GaiOgRuw",  # 김경국 TV
    "UC9Hlfu1tksuFbo2peL1ohnA",  # 최병묵의 FACT
    "UCKRBsup_e3Yxo3s4_nE8GVQ",  # 누리PD-TV
    "UC3iE8f8CdFFlbJm6tDifqNQ",  # 우파보이스
    "UCNybSCrV6IAh7CcpmoFMXXA",  # 박주현변호사TV
    "UC2B0mR5Xc61GnYKDyvrXZoA",  # 멸콩TV
    "UCmM7_1uOjGVVPL1nrfongjw",  # 학소
    "UCSZ1P1wJZdulg8pYMO9uXhg",  # 영상으로 보는 세상
    "UCKIQ3xPOi9tyJzcCMS5Uq3Q",  # 열린TV뉴스
    "UCv1THVoi-8Wwc_95wGZAizA",  # 정치한판
    "UC6_I-QpE75cn2iwzSEA9PRw",  # 주진우의 이슈해설
    "UC45hk7RSPPS_tezwn5rcL0A",  # 전옥현 안보정론TV
    "UCnacnZiUscX6jrVKu78Notg",  # 황교안TV
    "UCu-HTY6qT-zkRHp9jIiHbYg",  # 이영풍TV
    "UCfH-hmzHdzOwz6eTU_5Y3_A",  # 강작가
    "UCmUXqdG46YSBR6Gc9QGhZcg",  # [민경욱 TV]
    "UCh0J3pXsGV1JonYwraxl9aA",  # 제준뉴스
    "UCS7oapzKYfu85wuEtWWHk8A",  # 최국튜브
    "UC4DAnnB2ZAbZFjjgNd2SzgA",  # 고논
    "UChGHoVjZwiPAH2gOpHspR4Q",  # UNDER 73 STUDIO
    "UCTeyCGJfUixtN-pSivZMdpA",  # 김채환의 시사이다
    "UCQ21caARmy5hvLqvwIEBNUw",  # 도람뿌
    "UCPeWVzw9oQ-rZg49WY890Uw",  # 김PD의 사회와 경제 이야기
    "UCkW4UscEu3d4EIe73nCq40A",  # 김영윤TV_폴리티코 연구소
    "UCchCbWi1Mvibb6ddcZrrOHQ",  # 일요서울TV
    "UCpeD0mJAezRBRPieCBzV-Kg",  # 빨대포스트
]
