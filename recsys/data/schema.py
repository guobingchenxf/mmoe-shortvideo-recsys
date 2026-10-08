"""数据层 Schema：统一约定短视频推荐场景下各表的字段名与含义。


"""

from __future__ import annotations


class UserCol:
    """用户属性表字段。"""

    USER_ID = "user_id"
    AGE = "age"
    GENDER = "gender"                    # 0 女 / 1 男 / 2 未知
    CITY_LEVEL = "city_level"            # 城市等级 1~5
    ACTIVE_LEVEL = "active_level"        # 活跃度分档 0~3
    REGISTER_DAYS = "register_days"      # 注册天数（老用户/新用户）


class VideoCol:
    """视频内容表字段。"""

    VIDEO_ID = "video_id"
    AUTHOR_ID = "author_id"              # 创作者 id
    CATEGORY = "category"                # 一级分类 id
    GENRES = "genres"                    # 多值标签，用 GENRE_SEP 分隔（multi-hot 特征来源）
    DURATION_MS = "duration_ms"          # 视频时长（毫秒）
    UPLOAD_DAYS_AGO = "upload_days_ago"  # 距今上传天数（新鲜度）
    POPULARITY = "popularity"            # 热度分（历史曝光的对数，头部差异极大）


class InterCol:
    """曝光交互日志字段。"""

    USER_ID = "user_id"
    VIDEO_ID = "video_id"
    TIMESTAMP = "timestamp"

    # ---- 三个监督标签 ----
    CLICK = "click"                      # 是否点击（二分类）
    LIKE = "like"                        # 是否点赞（二分类）
    WATCH_RATIO = "watch_time_ratio"     # 完播率 0~1（回归）

    # ---- 上下文特征 ----
    HOUR = "hour"                        # 小时 0~23
    DOW = "day_of_week"                  # 星期 0~6
    DEVICE = "device"                    # 设备类型 0/1/2
    SOURCE = "source"                    # 流量来源页面 id


# ----------------------------------------------------------------------
# 多任务定义：顺序必须与模型 Tower 的输出顺序严格一致，改动需同步模型与评估代码
# ----------------------------------------------------------------------
LABEL_CLICK = InterCol.CLICK
LABEL_LIKE = InterCol.LIKE
LABEL_WATCH = InterCol.WATCH_RATIO
#: 三个监督目标（按塔的顺序）
MULTI_TASK_LABELS = (LABEL_CLICK, LABEL_LIKE, LABEL_WATCH)
#: 每个目标的任务类型：二分类用 BCE、回归用 MSE
LABEL_TASK_TYPES = {
    LABEL_CLICK: "binary",
    LABEL_LIKE: "binary",
    LABEL_WATCH: "regression",
}

# ----------------------------------------------------------------------
# 多值标签（genres）相关
# ----------------------------------------------------------------------
GENRE_SEP = "|"          # genres 字段的分隔符
N_TAG_VOCAB = 200        # 标签词表大小（生成时随机从该区间取值）

# ----------------------------------------------------------------------
# 文件名
# ----------------------------------------------------------------------
USERS_FILE = "users.csv"
VIDEOS_FILE = "videos.csv"
INTERACTIONS_FILE = "interactions.csv"
