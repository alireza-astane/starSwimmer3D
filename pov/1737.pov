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
    sphere { m*<0.9816571599236704,-1.5188523082083555e-18,0.7885057566439482>, 1 }        
    sphere {  m*<1.1500312077227974,9.207359715137114e-19,3.7837835954128387>, 1 }
    sphere {  m*<5.441492251082581,5.221221534873955e-18,-1.0763402779314568>, 1 }
    sphere {  m*<-3.9120501628115165,8.164965809277259,-2.2750064398223993>, 1}
    sphere { m*<-3.9120501628115165,-8.164965809277259,-2.275006439822403>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1500312077227974,9.207359715137114e-19,3.7837835954128387>, <0.9816571599236704,-1.5188523082083555e-18,0.7885057566439482>, 0.5 }
    cylinder { m*<5.441492251082581,5.221221534873955e-18,-1.0763402779314568>, <0.9816571599236704,-1.5188523082083555e-18,0.7885057566439482>, 0.5}
    cylinder { m*<-3.9120501628115165,8.164965809277259,-2.2750064398223993>, <0.9816571599236704,-1.5188523082083555e-18,0.7885057566439482>, 0.5 }
    cylinder {  m*<-3.9120501628115165,-8.164965809277259,-2.275006439822403>, <0.9816571599236704,-1.5188523082083555e-18,0.7885057566439482>, 0.5}

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
    sphere { m*<0.9816571599236704,-1.5188523082083555e-18,0.7885057566439482>, 1 }        
    sphere {  m*<1.1500312077227974,9.207359715137114e-19,3.7837835954128387>, 1 }
    sphere {  m*<5.441492251082581,5.221221534873955e-18,-1.0763402779314568>, 1 }
    sphere {  m*<-3.9120501628115165,8.164965809277259,-2.2750064398223993>, 1}
    sphere { m*<-3.9120501628115165,-8.164965809277259,-2.275006439822403>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1500312077227974,9.207359715137114e-19,3.7837835954128387>, <0.9816571599236704,-1.5188523082083555e-18,0.7885057566439482>, 0.5 }
    cylinder { m*<5.441492251082581,5.221221534873955e-18,-1.0763402779314568>, <0.9816571599236704,-1.5188523082083555e-18,0.7885057566439482>, 0.5}
    cylinder { m*<-3.9120501628115165,8.164965809277259,-2.2750064398223993>, <0.9816571599236704,-1.5188523082083555e-18,0.7885057566439482>, 0.5 }
    cylinder {  m*<-3.9120501628115165,-8.164965809277259,-2.275006439822403>, <0.9816571599236704,-1.5188523082083555e-18,0.7885057566439482>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    