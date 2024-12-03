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
    sphere { m*<-1.563031575419357,-0.18459149468052968,-1.0418012315295089>, 1 }        
    sphere {  m*<-0.10826780223877167,0.27719136447000514,8.840988988148254>, 1 }
    sphere {  m*<7.142008017565628,0.1128046145137736,-5.662932629902728>, 1 }
    sphere {  m*<-3.2393809188779743,2.1447514098000466,-1.9158730981822678>, 1}
    sphere { m*<-2.971593697840143,-2.7429405326038507,-1.7263268130196974>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.10826780223877167,0.27719136447000514,8.840988988148254>, <-1.563031575419357,-0.18459149468052968,-1.0418012315295089>, 0.5 }
    cylinder { m*<7.142008017565628,0.1128046145137736,-5.662932629902728>, <-1.563031575419357,-0.18459149468052968,-1.0418012315295089>, 0.5}
    cylinder { m*<-3.2393809188779743,2.1447514098000466,-1.9158730981822678>, <-1.563031575419357,-0.18459149468052968,-1.0418012315295089>, 0.5 }
    cylinder {  m*<-2.971593697840143,-2.7429405326038507,-1.7263268130196974>, <-1.563031575419357,-0.18459149468052968,-1.0418012315295089>, 0.5}

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
    sphere { m*<-1.563031575419357,-0.18459149468052968,-1.0418012315295089>, 1 }        
    sphere {  m*<-0.10826780223877167,0.27719136447000514,8.840988988148254>, 1 }
    sphere {  m*<7.142008017565628,0.1128046145137736,-5.662932629902728>, 1 }
    sphere {  m*<-3.2393809188779743,2.1447514098000466,-1.9158730981822678>, 1}
    sphere { m*<-2.971593697840143,-2.7429405326038507,-1.7263268130196974>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.10826780223877167,0.27719136447000514,8.840988988148254>, <-1.563031575419357,-0.18459149468052968,-1.0418012315295089>, 0.5 }
    cylinder { m*<7.142008017565628,0.1128046145137736,-5.662932629902728>, <-1.563031575419357,-0.18459149468052968,-1.0418012315295089>, 0.5}
    cylinder { m*<-3.2393809188779743,2.1447514098000466,-1.9158730981822678>, <-1.563031575419357,-0.18459149468052968,-1.0418012315295089>, 0.5 }
    cylinder {  m*<-2.971593697840143,-2.7429405326038507,-1.7263268130196974>, <-1.563031575419357,-0.18459149468052968,-1.0418012315295089>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    