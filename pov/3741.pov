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
    sphere { m*<-0.008047570974952234,0.1926656233660855,-0.13271316898988994>, 1 }        
    sphere {  m*<0.2326875337667395,0.32137570154641093,2.8548416021306617>, 1 }
    sphere {  m*<2.7266608230313114,0.2946995987524602,-1.361922694441077>, 1 }
    sphere {  m*<-1.6296629308678443,2.5211395677846884,-1.1066589344058624>, 1}
    sphere { m*<-2.1768802494207447,-3.907201609406349,-1.3893211363558247>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2326875337667395,0.32137570154641093,2.8548416021306617>, <-0.008047570974952234,0.1926656233660855,-0.13271316898988994>, 0.5 }
    cylinder { m*<2.7266608230313114,0.2946995987524602,-1.361922694441077>, <-0.008047570974952234,0.1926656233660855,-0.13271316898988994>, 0.5}
    cylinder { m*<-1.6296629308678443,2.5211395677846884,-1.1066589344058624>, <-0.008047570974952234,0.1926656233660855,-0.13271316898988994>, 0.5 }
    cylinder {  m*<-2.1768802494207447,-3.907201609406349,-1.3893211363558247>, <-0.008047570974952234,0.1926656233660855,-0.13271316898988994>, 0.5}

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
    sphere { m*<-0.008047570974952234,0.1926656233660855,-0.13271316898988994>, 1 }        
    sphere {  m*<0.2326875337667395,0.32137570154641093,2.8548416021306617>, 1 }
    sphere {  m*<2.7266608230313114,0.2946995987524602,-1.361922694441077>, 1 }
    sphere {  m*<-1.6296629308678443,2.5211395677846884,-1.1066589344058624>, 1}
    sphere { m*<-2.1768802494207447,-3.907201609406349,-1.3893211363558247>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2326875337667395,0.32137570154641093,2.8548416021306617>, <-0.008047570974952234,0.1926656233660855,-0.13271316898988994>, 0.5 }
    cylinder { m*<2.7266608230313114,0.2946995987524602,-1.361922694441077>, <-0.008047570974952234,0.1926656233660855,-0.13271316898988994>, 0.5}
    cylinder { m*<-1.6296629308678443,2.5211395677846884,-1.1066589344058624>, <-0.008047570974952234,0.1926656233660855,-0.13271316898988994>, 0.5 }
    cylinder {  m*<-2.1768802494207447,-3.907201609406349,-1.3893211363558247>, <-0.008047570974952234,0.1926656233660855,-0.13271316898988994>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    