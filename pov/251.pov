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
    sphere { m*<-5.42499500579363e-18,-1.7930936684739295e-19,0.3221386173651833>, 1 }        
    sphere {  m*<-9.082891754706933e-18,-2.0818798590953374e-18,8.558138617365195>, 1 }
    sphere {  m*<9.428090415820634,-1.7425085412163836e-18,-3.0111947159681485>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.0111947159681485>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.0111947159681485>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-9.082891754706933e-18,-2.0818798590953374e-18,8.558138617365195>, <-5.42499500579363e-18,-1.7930936684739295e-19,0.3221386173651833>, 0.5 }
    cylinder { m*<9.428090415820634,-1.7425085412163836e-18,-3.0111947159681485>, <-5.42499500579363e-18,-1.7930936684739295e-19,0.3221386173651833>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.0111947159681485>, <-5.42499500579363e-18,-1.7930936684739295e-19,0.3221386173651833>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.0111947159681485>, <-5.42499500579363e-18,-1.7930936684739295e-19,0.3221386173651833>, 0.5}

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
    sphere { m*<-5.42499500579363e-18,-1.7930936684739295e-19,0.3221386173651833>, 1 }        
    sphere {  m*<-9.082891754706933e-18,-2.0818798590953374e-18,8.558138617365195>, 1 }
    sphere {  m*<9.428090415820634,-1.7425085412163836e-18,-3.0111947159681485>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.0111947159681485>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.0111947159681485>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-9.082891754706933e-18,-2.0818798590953374e-18,8.558138617365195>, <-5.42499500579363e-18,-1.7930936684739295e-19,0.3221386173651833>, 0.5 }
    cylinder { m*<9.428090415820634,-1.7425085412163836e-18,-3.0111947159681485>, <-5.42499500579363e-18,-1.7930936684739295e-19,0.3221386173651833>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.0111947159681485>, <-5.42499500579363e-18,-1.7930936684739295e-19,0.3221386173651833>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.0111947159681485>, <-5.42499500579363e-18,-1.7930936684739295e-19,0.3221386173651833>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    