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
    sphere { m*<-0.2664862017576526,-0.1376958141015542,-1.6628302051302868>, 1 }        
    sphere {  m*<0.5320321406705377,0.2892355152225842,8.246888970676574>, 1 }
    sphere {  m*<2.4682221922486045,-0.03566183871518007,-2.8920397305814687>, 1 }
    sphere {  m*<-1.8881015616505425,2.190778130317045,-2.6367759705462555>, 1}
    sphere { m*<-1.6203143406127107,-2.6969138120868523,-2.4472296853836824>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5320321406705377,0.2892355152225842,8.246888970676574>, <-0.2664862017576526,-0.1376958141015542,-1.6628302051302868>, 0.5 }
    cylinder { m*<2.4682221922486045,-0.03566183871518007,-2.8920397305814687>, <-0.2664862017576526,-0.1376958141015542,-1.6628302051302868>, 0.5}
    cylinder { m*<-1.8881015616505425,2.190778130317045,-2.6367759705462555>, <-0.2664862017576526,-0.1376958141015542,-1.6628302051302868>, 0.5 }
    cylinder {  m*<-1.6203143406127107,-2.6969138120868523,-2.4472296853836824>, <-0.2664862017576526,-0.1376958141015542,-1.6628302051302868>, 0.5}

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
    sphere { m*<-0.2664862017576526,-0.1376958141015542,-1.6628302051302868>, 1 }        
    sphere {  m*<0.5320321406705377,0.2892355152225842,8.246888970676574>, 1 }
    sphere {  m*<2.4682221922486045,-0.03566183871518007,-2.8920397305814687>, 1 }
    sphere {  m*<-1.8881015616505425,2.190778130317045,-2.6367759705462555>, 1}
    sphere { m*<-1.6203143406127107,-2.6969138120868523,-2.4472296853836824>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5320321406705377,0.2892355152225842,8.246888970676574>, <-0.2664862017576526,-0.1376958141015542,-1.6628302051302868>, 0.5 }
    cylinder { m*<2.4682221922486045,-0.03566183871518007,-2.8920397305814687>, <-0.2664862017576526,-0.1376958141015542,-1.6628302051302868>, 0.5}
    cylinder { m*<-1.8881015616505425,2.190778130317045,-2.6367759705462555>, <-0.2664862017576526,-0.1376958141015542,-1.6628302051302868>, 0.5 }
    cylinder {  m*<-1.6203143406127107,-2.6969138120868523,-2.4472296853836824>, <-0.2664862017576526,-0.1376958141015542,-1.6628302051302868>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    