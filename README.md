1、GitHub新人的第一个项目，主要是用来熟悉模型训练的，看个乐和就行。
2、这一版的模型会根据你设置好的地址读取文件并进行模型分析，分析结果为5张图片和一些结果数据汇总。
3、这5张图片分别是混肴矩阵（Confusion Matrix）;特征相关性热力图（Correlation Heatmap）;泰坦尼克综合探索性分析图（Titanic EDA Combined）;
图 4：Error Rate by Sex & Pclass（分舱位 + 性别的模型预测错误率）和Feature Importance（特征重要性条形图）
4、本模型除了预测总体生还准确率以外还交叉验证平均一等舱男性错误率，因为在模型的不断优化过程中发现一等舱男性错误率达到了50%，这让我不得不对类样本进行模型的针对性优化。
5、如果懒得打开VSCode的话，你可以使用CMD命令终端进行运行，定位到文件所在位置，输入python main.py --weight 3.0 --oversample copy --copies 3 --calibrate True --cv 5，运行结果会同步显示在终端上
以下为具体参数信息
参数	          类型	      默认值	     说明
--weight	      float	       2.5       一等舱男性样本权重倍数
--oversample	  str	         copy	     过采样方式:smote / copy / none
--copies	      int	         2	       复制过采样时的复制倍数
--calibrate	    bool	       True	     是否进行概率校准
--cv	          int	         5	       交叉验证折数
--test_size	    float	       0.2	     测试集比例
--seed	        int	         42	       随机种子
--output_dir	  str	        自定义路径	 图片保存根目录
使用默认参数:python main.py

