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
    sphere { m*<-0.7643272557362052,-0.15716627119273463,-1.462216548349461>, 1 }        
    sphere {  m*<0.3188629442147904,0.2861679061007402,8.469037782737791>, 1 }
    sphere {  m*<4.460354189226235,0.0308399526126516,-3.9898292840072944>, 1 }
    sphere {  m*<-2.410251816410808,2.1716199210019553,-2.393710244993364>, 1}
    sphere { m*<-2.1424645953729766,-2.716072021401942,-2.204163959830793>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3188629442147904,0.2861679061007402,8.469037782737791>, <-0.7643272557362052,-0.15716627119273463,-1.462216548349461>, 0.5 }
    cylinder { m*<4.460354189226235,0.0308399526126516,-3.9898292840072944>, <-0.7643272557362052,-0.15716627119273463,-1.462216548349461>, 0.5}
    cylinder { m*<-2.410251816410808,2.1716199210019553,-2.393710244993364>, <-0.7643272557362052,-0.15716627119273463,-1.462216548349461>, 0.5 }
    cylinder {  m*<-2.1424645953729766,-2.716072021401942,-2.204163959830793>, <-0.7643272557362052,-0.15716627119273463,-1.462216548349461>, 0.5}

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
    sphere { m*<-0.7643272557362052,-0.15716627119273463,-1.462216548349461>, 1 }        
    sphere {  m*<0.3188629442147904,0.2861679061007402,8.469037782737791>, 1 }
    sphere {  m*<4.460354189226235,0.0308399526126516,-3.9898292840072944>, 1 }
    sphere {  m*<-2.410251816410808,2.1716199210019553,-2.393710244993364>, 1}
    sphere { m*<-2.1424645953729766,-2.716072021401942,-2.204163959830793>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3188629442147904,0.2861679061007402,8.469037782737791>, <-0.7643272557362052,-0.15716627119273463,-1.462216548349461>, 0.5 }
    cylinder { m*<4.460354189226235,0.0308399526126516,-3.9898292840072944>, <-0.7643272557362052,-0.15716627119273463,-1.462216548349461>, 0.5}
    cylinder { m*<-2.410251816410808,2.1716199210019553,-2.393710244993364>, <-0.7643272557362052,-0.15716627119273463,-1.462216548349461>, 0.5 }
    cylinder {  m*<-2.1424645953729766,-2.716072021401942,-2.204163959830793>, <-0.7643272557362052,-0.15716627119273463,-1.462216548349461>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    