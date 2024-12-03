#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.5139927299760448,1.1730617952155797,0.16977779331341664>, 1 }        
    sphere {  m*<0.7549261756874034,1.2884917850368245,3.157854182824826>, 1 }
    sphere {  m*<3.248173364749939,1.2884917850368238,-1.059428025665789>, 1 }
    sphere {  m*<-1.1257177533642029,3.5178437370704567,-0.7997144907392202>, 1}
    sphere { m*<-3.975935361547274,-7.358527756128826,-2.4842954851103434>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7549261756874034,1.2884917850368245,3.157854182824826>, <0.5139927299760448,1.1730617952155797,0.16977779331341664>, 0.5 }
    cylinder { m*<3.248173364749939,1.2884917850368238,-1.059428025665789>, <0.5139927299760448,1.1730617952155797,0.16977779331341664>, 0.5}
    cylinder { m*<-1.1257177533642029,3.5178437370704567,-0.7997144907392202>, <0.5139927299760448,1.1730617952155797,0.16977779331341664>, 0.5 }
    cylinder {  m*<-3.975935361547274,-7.358527756128826,-2.4842954851103434>, <0.5139927299760448,1.1730617952155797,0.16977779331341664>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.5139927299760448,1.1730617952155797,0.16977779331341664>, 1 }        
    sphere {  m*<0.7549261756874034,1.2884917850368245,3.157854182824826>, 1 }
    sphere {  m*<3.248173364749939,1.2884917850368238,-1.059428025665789>, 1 }
    sphere {  m*<-1.1257177533642029,3.5178437370704567,-0.7997144907392202>, 1}
    sphere { m*<-3.975935361547274,-7.358527756128826,-2.4842954851103434>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7549261756874034,1.2884917850368245,3.157854182824826>, <0.5139927299760448,1.1730617952155797,0.16977779331341664>, 0.5 }
    cylinder { m*<3.248173364749939,1.2884917850368238,-1.059428025665789>, <0.5139927299760448,1.1730617952155797,0.16977779331341664>, 0.5}
    cylinder { m*<-1.1257177533642029,3.5178437370704567,-0.7997144907392202>, <0.5139927299760448,1.1730617952155797,0.16977779331341664>, 0.5 }
    cylinder {  m*<-3.975935361547274,-7.358527756128826,-2.4842954851103434>, <0.5139927299760448,1.1730617952155797,0.16977779331341664>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    