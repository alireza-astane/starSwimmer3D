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
    sphere { m*<-0.5796966430934954,-0.783006330672875,-0.5180143037069187>, 1 }        
    sphere {  m*<0.8394708511066665,0.20693258320704233,9.331275793328231>, 1 }
    sphere {  m*<8.20725804942947,-0.0781596675852192,-5.2394016357457>, 1 }
    sphere {  m*<-6.688705144259521,6.444921706035415,-3.7485947325640936>, 1}
    sphere { m*<-3.2026688245871573,-6.495327586148181,-1.7326800633032295>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8394708511066665,0.20693258320704233,9.331275793328231>, <-0.5796966430934954,-0.783006330672875,-0.5180143037069187>, 0.5 }
    cylinder { m*<8.20725804942947,-0.0781596675852192,-5.2394016357457>, <-0.5796966430934954,-0.783006330672875,-0.5180143037069187>, 0.5}
    cylinder { m*<-6.688705144259521,6.444921706035415,-3.7485947325640936>, <-0.5796966430934954,-0.783006330672875,-0.5180143037069187>, 0.5 }
    cylinder {  m*<-3.2026688245871573,-6.495327586148181,-1.7326800633032295>, <-0.5796966430934954,-0.783006330672875,-0.5180143037069187>, 0.5}

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
    sphere { m*<-0.5796966430934954,-0.783006330672875,-0.5180143037069187>, 1 }        
    sphere {  m*<0.8394708511066665,0.20693258320704233,9.331275793328231>, 1 }
    sphere {  m*<8.20725804942947,-0.0781596675852192,-5.2394016357457>, 1 }
    sphere {  m*<-6.688705144259521,6.444921706035415,-3.7485947325640936>, 1}
    sphere { m*<-3.2026688245871573,-6.495327586148181,-1.7326800633032295>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8394708511066665,0.20693258320704233,9.331275793328231>, <-0.5796966430934954,-0.783006330672875,-0.5180143037069187>, 0.5 }
    cylinder { m*<8.20725804942947,-0.0781596675852192,-5.2394016357457>, <-0.5796966430934954,-0.783006330672875,-0.5180143037069187>, 0.5}
    cylinder { m*<-6.688705144259521,6.444921706035415,-3.7485947325640936>, <-0.5796966430934954,-0.783006330672875,-0.5180143037069187>, 0.5 }
    cylinder {  m*<-3.2026688245871573,-6.495327586148181,-1.7326800633032295>, <-0.5796966430934954,-0.783006330672875,-0.5180143037069187>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    