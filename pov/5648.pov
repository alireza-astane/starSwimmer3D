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
    sphere { m*<-1.1015483557675196,-0.1691454376385625,-1.295054467408386>, 1 }        
    sphere {  m*<0.1514690364165182,0.2826887398762507,8.615809091415437>, 1 }
    sphere {  m*<5.628189647524651,0.06748475503506429,-4.6938878547100575>, 1 }
    sphere {  m*<-2.761627180564945,2.1598748145443327,-2.200479015157337>, 1}
    sphere { m*<-2.4938399595271137,-2.7278171278595646,-2.0109327299947664>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1514690364165182,0.2826887398762507,8.615809091415437>, <-1.1015483557675196,-0.1691454376385625,-1.295054467408386>, 0.5 }
    cylinder { m*<5.628189647524651,0.06748475503506429,-4.6938878547100575>, <-1.1015483557675196,-0.1691454376385625,-1.295054467408386>, 0.5}
    cylinder { m*<-2.761627180564945,2.1598748145443327,-2.200479015157337>, <-1.1015483557675196,-0.1691454376385625,-1.295054467408386>, 0.5 }
    cylinder {  m*<-2.4938399595271137,-2.7278171278595646,-2.0109327299947664>, <-1.1015483557675196,-0.1691454376385625,-1.295054467408386>, 0.5}

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
    sphere { m*<-1.1015483557675196,-0.1691454376385625,-1.295054467408386>, 1 }        
    sphere {  m*<0.1514690364165182,0.2826887398762507,8.615809091415437>, 1 }
    sphere {  m*<5.628189647524651,0.06748475503506429,-4.6938878547100575>, 1 }
    sphere {  m*<-2.761627180564945,2.1598748145443327,-2.200479015157337>, 1}
    sphere { m*<-2.4938399595271137,-2.7278171278595646,-2.0109327299947664>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1514690364165182,0.2826887398762507,8.615809091415437>, <-1.1015483557675196,-0.1691454376385625,-1.295054467408386>, 0.5 }
    cylinder { m*<5.628189647524651,0.06748475503506429,-4.6938878547100575>, <-1.1015483557675196,-0.1691454376385625,-1.295054467408386>, 0.5}
    cylinder { m*<-2.761627180564945,2.1598748145443327,-2.200479015157337>, <-1.1015483557675196,-0.1691454376385625,-1.295054467408386>, 0.5 }
    cylinder {  m*<-2.4938399595271137,-2.7278171278595646,-2.0109327299947664>, <-1.1015483557675196,-0.1691454376385625,-1.295054467408386>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    