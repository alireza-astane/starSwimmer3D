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
    sphere { m*<1.0890886631505257,0.32669029804044514,0.5098090012447838>, 1 }        
    sphere {  m*<1.333174983329197,0.3521697034552966,3.499753174078223>, 1 }
    sphere {  m*<3.8264221723917315,0.3521697034552965,-0.7175290344123948>, 1 }
    sphere {  m*<-3.110109655220835,7.014552149774428,-1.9730304180890614>, 1}
    sphere { m*<-3.7763358326774186,-7.922317701784881,-2.3662688300988703>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.333174983329197,0.3521697034552966,3.499753174078223>, <1.0890886631505257,0.32669029804044514,0.5098090012447838>, 0.5 }
    cylinder { m*<3.8264221723917315,0.3521697034552965,-0.7175290344123948>, <1.0890886631505257,0.32669029804044514,0.5098090012447838>, 0.5}
    cylinder { m*<-3.110109655220835,7.014552149774428,-1.9730304180890614>, <1.0890886631505257,0.32669029804044514,0.5098090012447838>, 0.5 }
    cylinder {  m*<-3.7763358326774186,-7.922317701784881,-2.3662688300988703>, <1.0890886631505257,0.32669029804044514,0.5098090012447838>, 0.5}

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
    sphere { m*<1.0890886631505257,0.32669029804044514,0.5098090012447838>, 1 }        
    sphere {  m*<1.333174983329197,0.3521697034552966,3.499753174078223>, 1 }
    sphere {  m*<3.8264221723917315,0.3521697034552965,-0.7175290344123948>, 1 }
    sphere {  m*<-3.110109655220835,7.014552149774428,-1.9730304180890614>, 1}
    sphere { m*<-3.7763358326774186,-7.922317701784881,-2.3662688300988703>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.333174983329197,0.3521697034552966,3.499753174078223>, <1.0890886631505257,0.32669029804044514,0.5098090012447838>, 0.5 }
    cylinder { m*<3.8264221723917315,0.3521697034552965,-0.7175290344123948>, <1.0890886631505257,0.32669029804044514,0.5098090012447838>, 0.5}
    cylinder { m*<-3.110109655220835,7.014552149774428,-1.9730304180890614>, <1.0890886631505257,0.32669029804044514,0.5098090012447838>, 0.5 }
    cylinder {  m*<-3.7763358326774186,-7.922317701784881,-2.3662688300988703>, <1.0890886631505257,0.32669029804044514,0.5098090012447838>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    