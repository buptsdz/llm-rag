from django.urls import path

from . import views


app_name = "polls"  #指定当前命名空间
urlpatterns = [
    # 主页路径，对应 IndexView 类视图，名称为 "index"
    path("", views.IndexView.as_view(), name="index"),
    
    # 详细页面路径，使用整数参数 pk，对应 DetailView 类视图，名称为 "detail"
    path("<int:pk>/", views.DetailView.as_view(), name="detail"),
    
    # 结果页面路径，使用整数参数 pk，对应 ResultsView 类视图，名称为 "results"
    path("<int:pk>/results/", views.ResultsView.as_view(), name="results"),
    
    # 投票页面路径，使用整数参数 question_id，对应 vote 函数视图，名称为 "vote"
    path("<int:question_id>/vote/", views.vote, name="vote"),
]