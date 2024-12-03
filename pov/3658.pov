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
    sphere { m*<0.04374206621692184,0.29056650739015577,-0.10270658002197747>, 1 }        
    sphere {  m*<0.28447717095861347,0.4192765855704812,2.8848481910985733>, 1 }
    sphere {  m*<2.778450460223182,0.39260048277653026,-1.3319161054731632>, 1 }
    sphere {  m*<-1.5778732936759687,2.6190404518087584,-1.0766523454379486>, 1}
    sphere { m*<-2.3872819951204702,-4.304935944325428,-1.511226580063651>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.28447717095861347,0.4192765855704812,2.8848481910985733>, <0.04374206621692184,0.29056650739015577,-0.10270658002197747>, 0.5 }
    cylinder { m*<2.778450460223182,0.39260048277653026,-1.3319161054731632>, <0.04374206621692184,0.29056650739015577,-0.10270658002197747>, 0.5}
    cylinder { m*<-1.5778732936759687,2.6190404518087584,-1.0766523454379486>, <0.04374206621692184,0.29056650739015577,-0.10270658002197747>, 0.5 }
    cylinder {  m*<-2.3872819951204702,-4.304935944325428,-1.511226580063651>, <0.04374206621692184,0.29056650739015577,-0.10270658002197747>, 0.5}

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
    sphere { m*<0.04374206621692184,0.29056650739015577,-0.10270658002197747>, 1 }        
    sphere {  m*<0.28447717095861347,0.4192765855704812,2.8848481910985733>, 1 }
    sphere {  m*<2.778450460223182,0.39260048277653026,-1.3319161054731632>, 1 }
    sphere {  m*<-1.5778732936759687,2.6190404518087584,-1.0766523454379486>, 1}
    sphere { m*<-2.3872819951204702,-4.304935944325428,-1.511226580063651>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.28447717095861347,0.4192765855704812,2.8848481910985733>, <0.04374206621692184,0.29056650739015577,-0.10270658002197747>, 0.5 }
    cylinder { m*<2.778450460223182,0.39260048277653026,-1.3319161054731632>, <0.04374206621692184,0.29056650739015577,-0.10270658002197747>, 0.5}
    cylinder { m*<-1.5778732936759687,2.6190404518087584,-1.0766523454379486>, <0.04374206621692184,0.29056650739015577,-0.10270658002197747>, 0.5 }
    cylinder {  m*<-2.3872819951204702,-4.304935944325428,-1.511226580063651>, <0.04374206621692184,0.29056650739015577,-0.10270658002197747>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    