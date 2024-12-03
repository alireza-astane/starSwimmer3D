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
    sphere { m*<-0.24014114707679052,-0.12361031522395138,-1.3358845613211126>, 1 }        
    sphere {  m*<0.44715757696073843,0.24385695798087687,7.193584310228055>, 1 }
    sphere {  m*<2.4945672469294666,-0.02157633983757718,-2.565094086772294>, 1 }
    sphere {  m*<-1.8617565069696804,2.204863629194648,-2.3098303267370803>, 1}
    sphere { m*<-1.5939692859318486,-2.6828283132092494,-2.1202840415745077>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.44715757696073843,0.24385695798087687,7.193584310228055>, <-0.24014114707679052,-0.12361031522395138,-1.3358845613211126>, 0.5 }
    cylinder { m*<2.4945672469294666,-0.02157633983757718,-2.565094086772294>, <-0.24014114707679052,-0.12361031522395138,-1.3358845613211126>, 0.5}
    cylinder { m*<-1.8617565069696804,2.204863629194648,-2.3098303267370803>, <-0.24014114707679052,-0.12361031522395138,-1.3358845613211126>, 0.5 }
    cylinder {  m*<-1.5939692859318486,-2.6828283132092494,-2.1202840415745077>, <-0.24014114707679052,-0.12361031522395138,-1.3358845613211126>, 0.5}

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
    sphere { m*<-0.24014114707679052,-0.12361031522395138,-1.3358845613211126>, 1 }        
    sphere {  m*<0.44715757696073843,0.24385695798087687,7.193584310228055>, 1 }
    sphere {  m*<2.4945672469294666,-0.02157633983757718,-2.565094086772294>, 1 }
    sphere {  m*<-1.8617565069696804,2.204863629194648,-2.3098303267370803>, 1}
    sphere { m*<-1.5939692859318486,-2.6828283132092494,-2.1202840415745077>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.44715757696073843,0.24385695798087687,7.193584310228055>, <-0.24014114707679052,-0.12361031522395138,-1.3358845613211126>, 0.5 }
    cylinder { m*<2.4945672469294666,-0.02157633983757718,-2.565094086772294>, <-0.24014114707679052,-0.12361031522395138,-1.3358845613211126>, 0.5}
    cylinder { m*<-1.8617565069696804,2.204863629194648,-2.3098303267370803>, <-0.24014114707679052,-0.12361031522395138,-1.3358845613211126>, 0.5 }
    cylinder {  m*<-1.5939692859318486,-2.6828283132092494,-2.1202840415745077>, <-0.24014114707679052,-0.12361031522395138,-1.3358845613211126>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    