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
    sphere { m*<-0.1860182934772457,-0.09467329448651307,-0.6642127265403989>, 1 }        
    sphere {  m*<0.2552491535142748,0.14125227881802296,4.811975168923569>, 1 }
    sphere {  m*<2.5486901005290115,0.007360680899861097,-1.893422251991581>, 1 }
    sphere {  m*<-1.8076336533701358,2.2338006499320864,-1.6381584919563679>, 1}
    sphere { m*<-1.539846432332304,-2.653891292471811,-1.448612206793795>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2552491535142748,0.14125227881802296,4.811975168923569>, <-0.1860182934772457,-0.09467329448651307,-0.6642127265403989>, 0.5 }
    cylinder { m*<2.5486901005290115,0.007360680899861097,-1.893422251991581>, <-0.1860182934772457,-0.09467329448651307,-0.6642127265403989>, 0.5}
    cylinder { m*<-1.8076336533701358,2.2338006499320864,-1.6381584919563679>, <-0.1860182934772457,-0.09467329448651307,-0.6642127265403989>, 0.5 }
    cylinder {  m*<-1.539846432332304,-2.653891292471811,-1.448612206793795>, <-0.1860182934772457,-0.09467329448651307,-0.6642127265403989>, 0.5}

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
    sphere { m*<-0.1860182934772457,-0.09467329448651307,-0.6642127265403989>, 1 }        
    sphere {  m*<0.2552491535142748,0.14125227881802296,4.811975168923569>, 1 }
    sphere {  m*<2.5486901005290115,0.007360680899861097,-1.893422251991581>, 1 }
    sphere {  m*<-1.8076336533701358,2.2338006499320864,-1.6381584919563679>, 1}
    sphere { m*<-1.539846432332304,-2.653891292471811,-1.448612206793795>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2552491535142748,0.14125227881802296,4.811975168923569>, <-0.1860182934772457,-0.09467329448651307,-0.6642127265403989>, 0.5 }
    cylinder { m*<2.5486901005290115,0.007360680899861097,-1.893422251991581>, <-0.1860182934772457,-0.09467329448651307,-0.6642127265403989>, 0.5}
    cylinder { m*<-1.8076336533701358,2.2338006499320864,-1.6381584919563679>, <-0.1860182934772457,-0.09467329448651307,-0.6642127265403989>, 0.5 }
    cylinder {  m*<-1.539846432332304,-2.653891292471811,-1.448612206793795>, <-0.1860182934772457,-0.09467329448651307,-0.6642127265403989>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    