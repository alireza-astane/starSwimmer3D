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
    sphere { m*<-3.4319958904162636e-18,-2.4368449053619137e-18,0.5593220439708948>, 1 }        
    sphere {  m*<-4.122296013182681e-18,-4.7822126397542366e-18,7.458322043970916>, 1 }
    sphere {  m*<9.428090415820634,-1.744135391061539e-18,-2.7740112893624382>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.7740112893624382>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.7740112893624382>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.122296013182681e-18,-4.7822126397542366e-18,7.458322043970916>, <-3.4319958904162636e-18,-2.4368449053619137e-18,0.5593220439708948>, 0.5 }
    cylinder { m*<9.428090415820634,-1.744135391061539e-18,-2.7740112893624382>, <-3.4319958904162636e-18,-2.4368449053619137e-18,0.5593220439708948>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.7740112893624382>, <-3.4319958904162636e-18,-2.4368449053619137e-18,0.5593220439708948>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.7740112893624382>, <-3.4319958904162636e-18,-2.4368449053619137e-18,0.5593220439708948>, 0.5}

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
    sphere { m*<-3.4319958904162636e-18,-2.4368449053619137e-18,0.5593220439708948>, 1 }        
    sphere {  m*<-4.122296013182681e-18,-4.7822126397542366e-18,7.458322043970916>, 1 }
    sphere {  m*<9.428090415820634,-1.744135391061539e-18,-2.7740112893624382>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.7740112893624382>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.7740112893624382>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.122296013182681e-18,-4.7822126397542366e-18,7.458322043970916>, <-3.4319958904162636e-18,-2.4368449053619137e-18,0.5593220439708948>, 0.5 }
    cylinder { m*<9.428090415820634,-1.744135391061539e-18,-2.7740112893624382>, <-3.4319958904162636e-18,-2.4368449053619137e-18,0.5593220439708948>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.7740112893624382>, <-3.4319958904162636e-18,-2.4368449053619137e-18,0.5593220439708948>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.7740112893624382>, <-3.4319958904162636e-18,-2.4368449053619137e-18,0.5593220439708948>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    