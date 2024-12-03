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
    sphere { m*<0.24818585943678192,-2.4956795975724298e-18,1.1065096176880578>, 1 }        
    sphere {  m*<0.2817195973193621,-1.4518937573895508e-18,4.106323247579608>, 1 }
    sphere {  m*<8.468381550715884,5.236787586624928e-18,-1.9036333018425216>, 1 }
    sphere {  m*<-4.502826204859752,8.164965809277259,-2.1739234522907687>, 1}
    sphere { m*<-4.502826204859752,-8.164965809277259,-2.1739234522907713>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2817195973193621,-1.4518937573895508e-18,4.106323247579608>, <0.24818585943678192,-2.4956795975724298e-18,1.1065096176880578>, 0.5 }
    cylinder { m*<8.468381550715884,5.236787586624928e-18,-1.9036333018425216>, <0.24818585943678192,-2.4956795975724298e-18,1.1065096176880578>, 0.5}
    cylinder { m*<-4.502826204859752,8.164965809277259,-2.1739234522907687>, <0.24818585943678192,-2.4956795975724298e-18,1.1065096176880578>, 0.5 }
    cylinder {  m*<-4.502826204859752,-8.164965809277259,-2.1739234522907713>, <0.24818585943678192,-2.4956795975724298e-18,1.1065096176880578>, 0.5}

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
    sphere { m*<0.24818585943678192,-2.4956795975724298e-18,1.1065096176880578>, 1 }        
    sphere {  m*<0.2817195973193621,-1.4518937573895508e-18,4.106323247579608>, 1 }
    sphere {  m*<8.468381550715884,5.236787586624928e-18,-1.9036333018425216>, 1 }
    sphere {  m*<-4.502826204859752,8.164965809277259,-2.1739234522907687>, 1}
    sphere { m*<-4.502826204859752,-8.164965809277259,-2.1739234522907713>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2817195973193621,-1.4518937573895508e-18,4.106323247579608>, <0.24818585943678192,-2.4956795975724298e-18,1.1065096176880578>, 0.5 }
    cylinder { m*<8.468381550715884,5.236787586624928e-18,-1.9036333018425216>, <0.24818585943678192,-2.4956795975724298e-18,1.1065096176880578>, 0.5}
    cylinder { m*<-4.502826204859752,8.164965809277259,-2.1739234522907687>, <0.24818585943678192,-2.4956795975724298e-18,1.1065096176880578>, 0.5 }
    cylinder {  m*<-4.502826204859752,-8.164965809277259,-2.1739234522907713>, <0.24818585943678192,-2.4956795975724298e-18,1.1065096176880578>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    