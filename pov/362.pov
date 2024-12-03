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
    sphere { m*<-3.984761626692931e-18,1.2810428924643272e-19,0.4608079359877183>, 1 }        
    sphere {  m*<-6.6118262841949236e-18,-3.219902671874732e-18,7.919807935987734>, 1 }
    sphere {  m*<9.428090415820634,-3.0485948107968963e-18,-2.8725253973456133>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.8725253973456133>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.8725253973456133>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-6.6118262841949236e-18,-3.219902671874732e-18,7.919807935987734>, <-3.984761626692931e-18,1.2810428924643272e-19,0.4608079359877183>, 0.5 }
    cylinder { m*<9.428090415820634,-3.0485948107968963e-18,-2.8725253973456133>, <-3.984761626692931e-18,1.2810428924643272e-19,0.4608079359877183>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.8725253973456133>, <-3.984761626692931e-18,1.2810428924643272e-19,0.4608079359877183>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.8725253973456133>, <-3.984761626692931e-18,1.2810428924643272e-19,0.4608079359877183>, 0.5}

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
    sphere { m*<-3.984761626692931e-18,1.2810428924643272e-19,0.4608079359877183>, 1 }        
    sphere {  m*<-6.6118262841949236e-18,-3.219902671874732e-18,7.919807935987734>, 1 }
    sphere {  m*<9.428090415820634,-3.0485948107968963e-18,-2.8725253973456133>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.8725253973456133>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.8725253973456133>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-6.6118262841949236e-18,-3.219902671874732e-18,7.919807935987734>, <-3.984761626692931e-18,1.2810428924643272e-19,0.4608079359877183>, 0.5 }
    cylinder { m*<9.428090415820634,-3.0485948107968963e-18,-2.8725253973456133>, <-3.984761626692931e-18,1.2810428924643272e-19,0.4608079359877183>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.8725253973456133>, <-3.984761626692931e-18,1.2810428924643272e-19,0.4608079359877183>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.8725253973456133>, <-3.984761626692931e-18,1.2810428924643272e-19,0.4608079359877183>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    