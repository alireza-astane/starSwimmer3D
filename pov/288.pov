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
    sphere { m*<-5.789724885272624e-18,3.5012835950562665e-19,0.3686028473077202>, 1 }        
    sphere {  m*<-9.53986331376315e-18,-2.477980688715754e-18,8.345602847307731>, 1 }
    sphere {  m*<9.428090415820634,-1.9756356549500128e-18,-2.964730486025612>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.964730486025612>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.964730486025612>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-9.53986331376315e-18,-2.477980688715754e-18,8.345602847307731>, <-5.789724885272624e-18,3.5012835950562665e-19,0.3686028473077202>, 0.5 }
    cylinder { m*<9.428090415820634,-1.9756356549500128e-18,-2.964730486025612>, <-5.789724885272624e-18,3.5012835950562665e-19,0.3686028473077202>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.964730486025612>, <-5.789724885272624e-18,3.5012835950562665e-19,0.3686028473077202>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.964730486025612>, <-5.789724885272624e-18,3.5012835950562665e-19,0.3686028473077202>, 0.5}

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
    sphere { m*<-5.789724885272624e-18,3.5012835950562665e-19,0.3686028473077202>, 1 }        
    sphere {  m*<-9.53986331376315e-18,-2.477980688715754e-18,8.345602847307731>, 1 }
    sphere {  m*<9.428090415820634,-1.9756356549500128e-18,-2.964730486025612>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.964730486025612>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.964730486025612>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-9.53986331376315e-18,-2.477980688715754e-18,8.345602847307731>, <-5.789724885272624e-18,3.5012835950562665e-19,0.3686028473077202>, 0.5 }
    cylinder { m*<9.428090415820634,-1.9756356549500128e-18,-2.964730486025612>, <-5.789724885272624e-18,3.5012835950562665e-19,0.3686028473077202>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.964730486025612>, <-5.789724885272624e-18,3.5012835950562665e-19,0.3686028473077202>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.964730486025612>, <-5.789724885272624e-18,3.5012835950562665e-19,0.3686028473077202>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    