# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:42:22 2021

@author: 陳家瀅
"""




import pymysql
number=[]
# 資料庫設定
db_settings = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "",
    "db": "student_imformation",
    "charset": "utf8mb4"
}

def main():
    try:
        # 建立Connection物件
        conn = pymysql.connect(**db_settings)
        # 建立Cursor物件
        with conn.cursor() as cursor:
    
            command = "SELECT student_ID FROM student_imformation.demo;"       
            cursor.execute(command) # 執行指令  
            result = cursor.fetchall()# 取得所有資料
            for row in result:
                number.append(row[0])
            for i in number:
                for j in range(1,21):
                    sql1 = "UPDATE student_imformation.demo SET student_Temp"+str(j)+"=null WHERE student_ID='"+i+"';"
                    sql2 = "UPDATE student_imformation.demo SET student_Date"+str(j)+"=null WHERE student_ID='"+i+"';"
                    sql3 = "UPDATE student_imformation.demo SET student_check = '未簽到' WHERE student_ID='"+i+"';"
                    sql4 = "UPDATE student_imformation.student_password SET message = null WHERE account='"+i+"';"
                    cursor.execute(sql1)
                    cursor.execute(sql2)
                    cursor.execute(sql3)
                    cursor.execute(sql4)
                    conn.commit()#儲存變更
                    
                    
            
            
    except Exception as ex:
        print(ex)


if __name__ == '__main__':
    main()