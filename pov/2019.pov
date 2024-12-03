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
    sphere { m*<1.2639591464706115,0.027099624534138448,0.61320470862707>, 1 }        
    sphere {  m*<1.5082058667387592,0.029057603940673294,3.603244706278237>, 1 }
    sphere {  m*<4.001453055801296,0.0290576039406733,-0.6140375022123792>, 1 }
    sphere {  m*<-3.6440928498005936,8.071302129386801,-2.288764787274177>, 1}
    sphere { m*<-3.6974415246459063,-8.144378567872307,-2.3196172939219863>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.5082058667387592,0.029057603940673294,3.603244706278237>, <1.2639591464706115,0.027099624534138448,0.61320470862707>, 0.5 }
    cylinder { m*<4.001453055801296,0.0290576039406733,-0.6140375022123792>, <1.2639591464706115,0.027099624534138448,0.61320470862707>, 0.5}
    cylinder { m*<-3.6440928498005936,8.071302129386801,-2.288764787274177>, <1.2639591464706115,0.027099624534138448,0.61320470862707>, 0.5 }
    cylinder {  m*<-3.6974415246459063,-8.144378567872307,-2.3196172939219863>, <1.2639591464706115,0.027099624534138448,0.61320470862707>, 0.5}

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
    sphere { m*<1.2639591464706115,0.027099624534138448,0.61320470862707>, 1 }        
    sphere {  m*<1.5082058667387592,0.029057603940673294,3.603244706278237>, 1 }
    sphere {  m*<4.001453055801296,0.0290576039406733,-0.6140375022123792>, 1 }
    sphere {  m*<-3.6440928498005936,8.071302129386801,-2.288764787274177>, 1}
    sphere { m*<-3.6974415246459063,-8.144378567872307,-2.3196172939219863>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.5082058667387592,0.029057603940673294,3.603244706278237>, <1.2639591464706115,0.027099624534138448,0.61320470862707>, 0.5 }
    cylinder { m*<4.001453055801296,0.0290576039406733,-0.6140375022123792>, <1.2639591464706115,0.027099624534138448,0.61320470862707>, 0.5}
    cylinder { m*<-3.6440928498005936,8.071302129386801,-2.288764787274177>, <1.2639591464706115,0.027099624534138448,0.61320470862707>, 0.5 }
    cylinder {  m*<-3.6974415246459063,-8.144378567872307,-2.3196172939219863>, <1.2639591464706115,0.027099624534138448,0.61320470862707>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    