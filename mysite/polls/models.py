from django.utils import timezone
import datetime
from django.db import models

# 定义 Question 模型，它对应数据库中的一张表
class Question(models.Model):
    # 定义一个字符字段 question_text，最大长度为200个字符，用于存储问题的文本
    question_text = models.CharField(max_length=200)
    
    # 定义一个日期时间字段 pub_date，用于存储问题的发布时间
    pub_date = models.DateTimeField("date published")

    # 定义模型对象的字符串表示形式，便于在管理界面和 shell 中查看对象
    def __str__(self):
        return self.question_text
    
    def was_published_recently(self):
        now = timezone.now()
        return now - datetime.timedelta(days=1) <= self.pub_date <= now

# 定义 Choice 模型，它也对应数据库中的一张表
class Choice(models.Model):
    # 定义一个外键字段 question，关联到 Question 模型，并且在关联的 Question 被删除时，级联删除相关的 Choice
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    
    # 定义一个字符字段 choice_text，最大长度为200个字符，用于存储选项的文本
    choice_text = models.CharField(max_length=200)
    
    # 定义一个整数字段 votes，默认值为0，用于存储投票数
    votes = models.IntegerField(default=0)

    # 定义模型对象的字符串表示形式，便于在管理界面和 shell 中查看对象
    def __str__(self):
        return self.choice_text
