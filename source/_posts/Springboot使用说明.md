---
title: Springboot使用说明
date: 2018-10-25 13:29:09
tags:
- springboot
categories: other
description: Springboot 技术简单使用说明
---

# 1. 新建Spring Boot项目
1. 打开Myeclipse,在左侧项目区域右键，选择1`new`-> `web project`。填写项目名称并勾选
- `Add Maven support`，选择next，在最后勾选
- `Standard Maven JEE project structure`，选择完成
2. 将sport项目中的`pom.xml`内容复制覆盖新建项目的`pom.xml`，修改第一行的项目名称和版本，以及`spring-boot-maven-plugin/Application`信息。
3. 将`Application.java`,`application.properties`文件复制到相应位置。
4. 进行项目包分层。`bean`,`dao`,`service`,`contol`等
5. 编写测试方法。
```java
@RestController
public class Default{

    @RequestMapping(value="/")
    public String defaultMethod(){
        return "{'name':'Jack', 'gender':'female'}";
    }
}
```

# 2. Mybatis generator使用
1. 将`mabatis-generator.xml`复制到对应位置，并修改其内容，填写JDBC驱动包、待生成路径、数据库对应表等信息。
2. 修改`pom.xml`，取消`mybatis-generator`的注释。
3. 右键项目，选择 `Run As` - `Maven build`, 在`Goals`填写`mabatis-generator:generate`,选择执行。
4. Maven自动下载包并执行。

# 3. Spring Boot Jar包生成
选择`Run As`-`Maven install`会自动生成可运行Jar包。
