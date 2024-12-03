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
    sphere { m*<-0.18295320222957132,-0.09303453002571163,-0.6261745351817477>, 1 }        
    sphere {  m*<0.2431479331632228,0.13478230835346405,4.661797409701629>, 1 }
    sphere {  m*<2.5517551917766856,0.00899944536066255,-1.85538406063293>, 1 }
    sphere {  m*<-1.8045685621224614,2.2354394143928875,-1.6001203005977167>, 1}
    sphere { m*<-1.5367813410846296,-2.65225252801101,-1.4105740154351438>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2431479331632228,0.13478230835346405,4.661797409701629>, <-0.18295320222957132,-0.09303453002571163,-0.6261745351817477>, 0.5 }
    cylinder { m*<2.5517551917766856,0.00899944536066255,-1.85538406063293>, <-0.18295320222957132,-0.09303453002571163,-0.6261745351817477>, 0.5}
    cylinder { m*<-1.8045685621224614,2.2354394143928875,-1.6001203005977167>, <-0.18295320222957132,-0.09303453002571163,-0.6261745351817477>, 0.5 }
    cylinder {  m*<-1.5367813410846296,-2.65225252801101,-1.4105740154351438>, <-0.18295320222957132,-0.09303453002571163,-0.6261745351817477>, 0.5}

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
    sphere { m*<-0.18295320222957132,-0.09303453002571163,-0.6261745351817477>, 1 }        
    sphere {  m*<0.2431479331632228,0.13478230835346405,4.661797409701629>, 1 }
    sphere {  m*<2.5517551917766856,0.00899944536066255,-1.85538406063293>, 1 }
    sphere {  m*<-1.8045685621224614,2.2354394143928875,-1.6001203005977167>, 1}
    sphere { m*<-1.5367813410846296,-2.65225252801101,-1.4105740154351438>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2431479331632228,0.13478230835346405,4.661797409701629>, <-0.18295320222957132,-0.09303453002571163,-0.6261745351817477>, 0.5 }
    cylinder { m*<2.5517551917766856,0.00899944536066255,-1.85538406063293>, <-0.18295320222957132,-0.09303453002571163,-0.6261745351817477>, 0.5}
    cylinder { m*<-1.8045685621224614,2.2354394143928875,-1.6001203005977167>, <-0.18295320222957132,-0.09303453002571163,-0.6261745351817477>, 0.5 }
    cylinder {  m*<-1.5367813410846296,-2.65225252801101,-1.4105740154351438>, <-0.18295320222957132,-0.09303453002571163,-0.6261745351817477>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    