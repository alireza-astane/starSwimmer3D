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
    sphere { m*<-0.672774382785207,-0.9857114735653123,-0.561117446081749>, 1 }        
    sphere {  m*<0.7463931114149557,0.004227440314605646,9.288172650953404>, 1 }
    sphere {  m*<8.114180309737753,-0.2808648104776572,-5.28250477812053>, 1 }
    sphere {  m*<-6.7817828839512355,6.242216563142989,-3.791697874938925>, 1}
    sphere { m*<-2.7450944849399916,-5.498820019362577,-1.5207830837009526>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7463931114149557,0.004227440314605646,9.288172650953404>, <-0.672774382785207,-0.9857114735653123,-0.561117446081749>, 0.5 }
    cylinder { m*<8.114180309737753,-0.2808648104776572,-5.28250477812053>, <-0.672774382785207,-0.9857114735653123,-0.561117446081749>, 0.5}
    cylinder { m*<-6.7817828839512355,6.242216563142989,-3.791697874938925>, <-0.672774382785207,-0.9857114735653123,-0.561117446081749>, 0.5 }
    cylinder {  m*<-2.7450944849399916,-5.498820019362577,-1.5207830837009526>, <-0.672774382785207,-0.9857114735653123,-0.561117446081749>, 0.5}

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
    sphere { m*<-0.672774382785207,-0.9857114735653123,-0.561117446081749>, 1 }        
    sphere {  m*<0.7463931114149557,0.004227440314605646,9.288172650953404>, 1 }
    sphere {  m*<8.114180309737753,-0.2808648104776572,-5.28250477812053>, 1 }
    sphere {  m*<-6.7817828839512355,6.242216563142989,-3.791697874938925>, 1}
    sphere { m*<-2.7450944849399916,-5.498820019362577,-1.5207830837009526>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7463931114149557,0.004227440314605646,9.288172650953404>, <-0.672774382785207,-0.9857114735653123,-0.561117446081749>, 0.5 }
    cylinder { m*<8.114180309737753,-0.2808648104776572,-5.28250477812053>, <-0.672774382785207,-0.9857114735653123,-0.561117446081749>, 0.5}
    cylinder { m*<-6.7817828839512355,6.242216563142989,-3.791697874938925>, <-0.672774382785207,-0.9857114735653123,-0.561117446081749>, 0.5 }
    cylinder {  m*<-2.7450944849399916,-5.498820019362577,-1.5207830837009526>, <-0.672774382785207,-0.9857114735653123,-0.561117446081749>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    