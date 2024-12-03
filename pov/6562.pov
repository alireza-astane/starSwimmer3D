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
    sphere { m*<-1.188816098260366,-0.7998928613202206,-0.8188888468206679>, 1 }        
    sphere {  m*<0.2538050965671901,-0.07506666346057549,9.05000110320524>, 1 }
    sphere {  m*<7.60915653456716,-0.16398693945493256,-5.529492186840107>, 1 }
    sphere {  m*<-5.325416828721583,4.35557037416482,-2.93739852358501>, 1}
    sphere { m*<-2.4729388902486598,-3.4362243486030715,-1.450646193137758>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2538050965671901,-0.07506666346057549,9.05000110320524>, <-1.188816098260366,-0.7998928613202206,-0.8188888468206679>, 0.5 }
    cylinder { m*<7.60915653456716,-0.16398693945493256,-5.529492186840107>, <-1.188816098260366,-0.7998928613202206,-0.8188888468206679>, 0.5}
    cylinder { m*<-5.325416828721583,4.35557037416482,-2.93739852358501>, <-1.188816098260366,-0.7998928613202206,-0.8188888468206679>, 0.5 }
    cylinder {  m*<-2.4729388902486598,-3.4362243486030715,-1.450646193137758>, <-1.188816098260366,-0.7998928613202206,-0.8188888468206679>, 0.5}

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
    sphere { m*<-1.188816098260366,-0.7998928613202206,-0.8188888468206679>, 1 }        
    sphere {  m*<0.2538050965671901,-0.07506666346057549,9.05000110320524>, 1 }
    sphere {  m*<7.60915653456716,-0.16398693945493256,-5.529492186840107>, 1 }
    sphere {  m*<-5.325416828721583,4.35557037416482,-2.93739852358501>, 1}
    sphere { m*<-2.4729388902486598,-3.4362243486030715,-1.450646193137758>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2538050965671901,-0.07506666346057549,9.05000110320524>, <-1.188816098260366,-0.7998928613202206,-0.8188888468206679>, 0.5 }
    cylinder { m*<7.60915653456716,-0.16398693945493256,-5.529492186840107>, <-1.188816098260366,-0.7998928613202206,-0.8188888468206679>, 0.5}
    cylinder { m*<-5.325416828721583,4.35557037416482,-2.93739852358501>, <-1.188816098260366,-0.7998928613202206,-0.8188888468206679>, 0.5 }
    cylinder {  m*<-2.4729388902486598,-3.4362243486030715,-1.450646193137758>, <-1.188816098260366,-0.7998928613202206,-0.8188888468206679>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    