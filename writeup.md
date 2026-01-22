1. Problem (unicode1)
   
   (a). chr(0)="\x00"即空字符

   (b). 打印即为空字符,repr形式是"\x00"

   (c). 不使用print打印则用"\x00"表示,打印出来则为空
        
        
        >>> "this is a test" + chr(0) + "string"
            'this is a test\x00string'
        >>> print("this is a test" + chr(0) + "string")
            this is a teststring

2. Problem (unicode2)
   
   (a). utf8编码当中的0的个数显著少于utf16和utf32,换言之后两者之编码结果过于稀疏了。

   (b). 并非每个字节都是可以有意义的转写,譬如以"你好"为例代替"hello"则见函数不可以运行。 