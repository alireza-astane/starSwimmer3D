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
    sphere { m*<-0.6430003250654067,-0.1526877751706816,-1.5180000375872738>, 1 }        
    sphere {  m*<0.37459318550496595,0.28733272153195943,8.420341491903772>, 1 }
    sphere {  m*<4.016907106736786,0.016456451304410813,-3.7346089142652006>, 1 }
    sphere {  m*<-2.283346251709818,2.1760171482902897,-2.459484447663967>, 1}
    sphere { m*<-2.0155590306719864,-2.7116747941136077,-2.2699381625013966>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.37459318550496595,0.28733272153195943,8.420341491903772>, <-0.6430003250654067,-0.1526877751706816,-1.5180000375872738>, 0.5 }
    cylinder { m*<4.016907106736786,0.016456451304410813,-3.7346089142652006>, <-0.6430003250654067,-0.1526877751706816,-1.5180000375872738>, 0.5}
    cylinder { m*<-2.283346251709818,2.1760171482902897,-2.459484447663967>, <-0.6430003250654067,-0.1526877751706816,-1.5180000375872738>, 0.5 }
    cylinder {  m*<-2.0155590306719864,-2.7116747941136077,-2.2699381625013966>, <-0.6430003250654067,-0.1526877751706816,-1.5180000375872738>, 0.5}

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
    sphere { m*<-0.6430003250654067,-0.1526877751706816,-1.5180000375872738>, 1 }        
    sphere {  m*<0.37459318550496595,0.28733272153195943,8.420341491903772>, 1 }
    sphere {  m*<4.016907106736786,0.016456451304410813,-3.7346089142652006>, 1 }
    sphere {  m*<-2.283346251709818,2.1760171482902897,-2.459484447663967>, 1}
    sphere { m*<-2.0155590306719864,-2.7116747941136077,-2.2699381625013966>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.37459318550496595,0.28733272153195943,8.420341491903772>, <-0.6430003250654067,-0.1526877751706816,-1.5180000375872738>, 0.5 }
    cylinder { m*<4.016907106736786,0.016456451304410813,-3.7346089142652006>, <-0.6430003250654067,-0.1526877751706816,-1.5180000375872738>, 0.5}
    cylinder { m*<-2.283346251709818,2.1760171482902897,-2.459484447663967>, <-0.6430003250654067,-0.1526877751706816,-1.5180000375872738>, 0.5 }
    cylinder {  m*<-2.0155590306719864,-2.7116747941136077,-2.2699381625013966>, <-0.6430003250654067,-0.1526877751706816,-1.5180000375872738>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    