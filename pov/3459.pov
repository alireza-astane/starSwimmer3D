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
    sphere { m*<0.17624623482515162,0.5410466410037515,-0.025934500904458868>, 1 }        
    sphere {  m*<0.41698133956684325,0.6697567191840771,2.961620270216092>, 1 }
    sphere {  m*<2.910954628831409,0.643080616390126,-1.2551440263556435>, 1 }
    sphere {  m*<-1.4453691250677387,2.869520585422353,-0.9998802663204293>, 1}
    sphere { m*<-2.883405358987276,-5.242786034443009,-1.7986773262771263>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.41698133956684325,0.6697567191840771,2.961620270216092>, <0.17624623482515162,0.5410466410037515,-0.025934500904458868>, 0.5 }
    cylinder { m*<2.910954628831409,0.643080616390126,-1.2551440263556435>, <0.17624623482515162,0.5410466410037515,-0.025934500904458868>, 0.5}
    cylinder { m*<-1.4453691250677387,2.869520585422353,-0.9998802663204293>, <0.17624623482515162,0.5410466410037515,-0.025934500904458868>, 0.5 }
    cylinder {  m*<-2.883405358987276,-5.242786034443009,-1.7986773262771263>, <0.17624623482515162,0.5410466410037515,-0.025934500904458868>, 0.5}

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
    sphere { m*<0.17624623482515162,0.5410466410037515,-0.025934500904458868>, 1 }        
    sphere {  m*<0.41698133956684325,0.6697567191840771,2.961620270216092>, 1 }
    sphere {  m*<2.910954628831409,0.643080616390126,-1.2551440263556435>, 1 }
    sphere {  m*<-1.4453691250677387,2.869520585422353,-0.9998802663204293>, 1}
    sphere { m*<-2.883405358987276,-5.242786034443009,-1.7986773262771263>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.41698133956684325,0.6697567191840771,2.961620270216092>, <0.17624623482515162,0.5410466410037515,-0.025934500904458868>, 0.5 }
    cylinder { m*<2.910954628831409,0.643080616390126,-1.2551440263556435>, <0.17624623482515162,0.5410466410037515,-0.025934500904458868>, 0.5}
    cylinder { m*<-1.4453691250677387,2.869520585422353,-0.9998802663204293>, <0.17624623482515162,0.5410466410037515,-0.025934500904458868>, 0.5 }
    cylinder {  m*<-2.883405358987276,-5.242786034443009,-1.7986773262771263>, <0.17624623482515162,0.5410466410037515,-0.025934500904458868>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    