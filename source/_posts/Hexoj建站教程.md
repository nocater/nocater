---
title: Hexo Blog 教程
date: 2018-10-24 18:14:18
tags: 
- Hexo 
- Github Page
categories: other
description: Hexo建站流程说明
---

# 步骤 #
## 初始化 ##
``` bash
$ hexo init [folder]
```

## 新建文章 ##
``` bash
$ hexo new [layout] <title>
```
如果没有设置`layout`，默认使用config.yml中的`defaultlayout`参数代替。如果标题包含空格，请使用引号括起来。

## 生成静态网页 ##
``` bash
$ hexo generate
```
可以简写成
``` bash
$ hexo g
```

## 启动服务 ##
``` bash
$ hexo server
```
启动服务器。默认情况下，访问网址为： `http://localhost:4000/`。

## 部署 ##
``` bash
$ hexo deploy
```
**该命令请在git中执行。**  部署网站，可以简写为
```
$ hexo d
```

## 修改站点位置文件 ##
文件位于项目根路径下`_config.yml`文件，
``` bash
deploy:
  type: git
  repo: "仓库路径"
  branch: master
```
这里注意`:`后一定要有一个空格，否则部署无反应。
  
## 安装Hexo-git ##
``` bash
$ npm install hexo-deployer-git --save
```
之后使用发布命令即可。

## 更换主题 ##
1. 下载主题：
``` bash
$ git clone https://github.com/iissnan/hexo-theme-next themes/next
```
2. 配置主题：
修改站点默认主题
``` 
theme: next
```
可以选择修改样式.打开主题配置文件`next/_config.yml`，选择以下即可：
``` 
#scheme: Muse
#scheme: Mist
scheme: Pisces
#scheme: Gemini
```

## Profile信息修改 ##
打开站点配置文件`_config.yml`，修改对应文字即可：
```
# Site
title: Hexo
subtitle:
description:
author: John Doe
language:
timezone:
```
**Next的使用以后再调整**

## 文章标签 ##
首先创建tag页面：
``` bash
$ hexo new page tags
```
修改`source/tags/index.md`文件，添加`type: "tags"`：
```
---
title: tags
date: 2018-10-24 21:25:58
type: "tags"
---
```
修改模板文件`scaffolds/post.md`,添加一行'tags:':
```
---
title: {{ title }}
date: {{ date }}
tags:
categories:
description:
---
```
给文章添加tags,在文章开头填写tags，格式如下：
```
tags: 
- Hexo 
- Github Page
```

## 文章分类 ##
同创建标签步骤基本一致，首先创建分类页面：
``` bash
$ hexo new page categories
```
然后修改`source/tags/index.md`文件，添加`type: "tags"`：
```
---
title: tags
date: 2018-10-24 21:25:58
type: "categories"
---
```
最后在文章添加分类：
```
---
title: Hexo教程
date: 2018-10-24 18:14:18
tags: 
- Hexo 
- Github Page
categories: other
description: Hexo建站流程说明
---
```

## 阅读统计 ##
阅读次数统计（LeanCloud） 由 Doublemine 贡献
请查看[为NexT主题添加文章阅读量统计功能](https://notes.wanghao.work/2015-10-21-%E4%B8%BANexT%E4%B8%BB%E9%A2%98%E6%B7%BB%E5%8A%A0%E6%96%87%E7%AB%A0%E9%98%85%E8%AF%BB%E9%87%8F%E7%BB%9F%E8%AE%A1%E5%8A%9F%E8%83%BD.html#%E9%85%8D%E7%BD%AELeanCloud)

## Latex注意事项 ##
- 先写下标，再写上标，否则无法编译。`$X_1^2$`
- `{}`的转义不是`\{\}`，而是`\\{\\}`:$\\{\\}$
- 公式换行`\\`转义为`\\\\`:
$$
\begin{cases}
1 \\\\
2 \\\\
3 \\\\
\end{cases}
$$
- 表头前要有一行空白，否则编译失败
- `*` 需要转义为`\*`：$\*$
- `<t>`需要转义`<t\>`:$x^{<t\>}$

**请注意**：  
`mathjax`的编译，请选择不同cdn。
``` 
mathjax:
  enable: true
  per_page: false
  # 本地Latex编译 cdn: //cdn.bootcss.com/mathjax/2.7.1/latest.js?config=TeX-AMS-MML_HTMLorMML
  # GithubLatex编译 cdn: //cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML
```

## 发布部署说明 ##
- `hexo` 分支:hexo笔记源代码
- `master` 分支:hexo笔记访问分支
- `jekyll`分支:之前博客文章备份
源码文件夹一直处于`hexo`分支，直接修改博客，然后修改。源文件直接`commit`到`hexo`分支。  
部署直接git cmd使用
``` bash
hexo clean
hexo g
hexo d
```
`hexo`已追踪文件及文件夹:
- `.gitignore`
- `_config.yml`
- `source/`
- `theme/next/_config.yml`
- `scaffolds/`


# 参考 #

[Hexo官方文档](https://hexo.io/zh-cn/docs/commands)  
[NexT官方教程](https://theme-next.iissnan.com/getting-started.html)