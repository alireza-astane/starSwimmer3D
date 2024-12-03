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
    sphere { m*<-2.0820141107978413e-18,-5.326462802092957e-18,1.102232576425055>, 1 }        
    sphere {  m*<-3.896503255581972e-18,-4.576610883575185e-18,4.7252325764250935>, 1 }
    sphere {  m*<9.428090415820634,2.087038389182456e-19,-2.231100756908278>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.231100756908278>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.231100756908278>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.896503255581972e-18,-4.576610883575185e-18,4.7252325764250935>, <-2.0820141107978413e-18,-5.326462802092957e-18,1.102232576425055>, 0.5 }
    cylinder { m*<9.428090415820634,2.087038389182456e-19,-2.231100756908278>, <-2.0820141107978413e-18,-5.326462802092957e-18,1.102232576425055>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.231100756908278>, <-2.0820141107978413e-18,-5.326462802092957e-18,1.102232576425055>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.231100756908278>, <-2.0820141107978413e-18,-5.326462802092957e-18,1.102232576425055>, 0.5}

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
    sphere { m*<-2.0820141107978413e-18,-5.326462802092957e-18,1.102232576425055>, 1 }        
    sphere {  m*<-3.896503255581972e-18,-4.576610883575185e-18,4.7252325764250935>, 1 }
    sphere {  m*<9.428090415820634,2.087038389182456e-19,-2.231100756908278>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.231100756908278>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.231100756908278>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.896503255581972e-18,-4.576610883575185e-18,4.7252325764250935>, <-2.0820141107978413e-18,-5.326462802092957e-18,1.102232576425055>, 0.5 }
    cylinder { m*<9.428090415820634,2.087038389182456e-19,-2.231100756908278>, <-2.0820141107978413e-18,-5.326462802092957e-18,1.102232576425055>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.231100756908278>, <-2.0820141107978413e-18,-5.326462802092957e-18,1.102232576425055>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.231100756908278>, <-2.0820141107978413e-18,-5.326462802092957e-18,1.102232576425055>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    