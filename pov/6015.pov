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
    sphere { m*<-1.581907858995764,-0.20208300112319882,-1.020865362591431>, 1 }        
    sphere {  m*<-0.11604742778485799,0.23591931411107386,8.861502053152481>, 1 }
    sphere {  m*<7.239304010215105,0.14699903811671589,-5.717991236892882>, 1 }
    sphere {  m*<-3.3312039819437187,2.2102682254432273,-1.9181650018680017>, 1}
    sphere { m*<-2.981682679818734,-2.763732950810645,-1.7114280441189593>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.11604742778485799,0.23591931411107386,8.861502053152481>, <-1.581907858995764,-0.20208300112319882,-1.020865362591431>, 0.5 }
    cylinder { m*<7.239304010215105,0.14699903811671589,-5.717991236892882>, <-1.581907858995764,-0.20208300112319882,-1.020865362591431>, 0.5}
    cylinder { m*<-3.3312039819437187,2.2102682254432273,-1.9181650018680017>, <-1.581907858995764,-0.20208300112319882,-1.020865362591431>, 0.5 }
    cylinder {  m*<-2.981682679818734,-2.763732950810645,-1.7114280441189593>, <-1.581907858995764,-0.20208300112319882,-1.020865362591431>, 0.5}

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
    sphere { m*<-1.581907858995764,-0.20208300112319882,-1.020865362591431>, 1 }        
    sphere {  m*<-0.11604742778485799,0.23591931411107386,8.861502053152481>, 1 }
    sphere {  m*<7.239304010215105,0.14699903811671589,-5.717991236892882>, 1 }
    sphere {  m*<-3.3312039819437187,2.2102682254432273,-1.9181650018680017>, 1}
    sphere { m*<-2.981682679818734,-2.763732950810645,-1.7114280441189593>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.11604742778485799,0.23591931411107386,8.861502053152481>, <-1.581907858995764,-0.20208300112319882,-1.020865362591431>, 0.5 }
    cylinder { m*<7.239304010215105,0.14699903811671589,-5.717991236892882>, <-1.581907858995764,-0.20208300112319882,-1.020865362591431>, 0.5}
    cylinder { m*<-3.3312039819437187,2.2102682254432273,-1.9181650018680017>, <-1.581907858995764,-0.20208300112319882,-1.020865362591431>, 0.5 }
    cylinder {  m*<-2.981682679818734,-2.763732950810645,-1.7114280441189593>, <-1.581907858995764,-0.20208300112319882,-1.020865362591431>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    