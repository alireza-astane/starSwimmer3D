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
    sphere { m*<0.05206218099399751,0.30629449152025623,-0.09788595802463446>, 1 }        
    sphere {  m*<0.2927972857356892,0.43500456970058166,2.8896688130959163>, 1 }
    sphere {  m*<2.7867705750002574,0.4083284669066307,-1.3270954834758197>, 1 }
    sphere {  m*<-1.5695531788988935,2.634768435938858,-1.0718317234406052>, 1}
    sphere { m*<-2.420028000555332,-4.366837572800881,-1.530199408967328>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2927972857356892,0.43500456970058166,2.8896688130959163>, <0.05206218099399751,0.30629449152025623,-0.09788595802463446>, 0.5 }
    cylinder { m*<2.7867705750002574,0.4083284669066307,-1.3270954834758197>, <0.05206218099399751,0.30629449152025623,-0.09788595802463446>, 0.5}
    cylinder { m*<-1.5695531788988935,2.634768435938858,-1.0718317234406052>, <0.05206218099399751,0.30629449152025623,-0.09788595802463446>, 0.5 }
    cylinder {  m*<-2.420028000555332,-4.366837572800881,-1.530199408967328>, <0.05206218099399751,0.30629449152025623,-0.09788595802463446>, 0.5}

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
    sphere { m*<0.05206218099399751,0.30629449152025623,-0.09788595802463446>, 1 }        
    sphere {  m*<0.2927972857356892,0.43500456970058166,2.8896688130959163>, 1 }
    sphere {  m*<2.7867705750002574,0.4083284669066307,-1.3270954834758197>, 1 }
    sphere {  m*<-1.5695531788988935,2.634768435938858,-1.0718317234406052>, 1}
    sphere { m*<-2.420028000555332,-4.366837572800881,-1.530199408967328>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2927972857356892,0.43500456970058166,2.8896688130959163>, <0.05206218099399751,0.30629449152025623,-0.09788595802463446>, 0.5 }
    cylinder { m*<2.7867705750002574,0.4083284669066307,-1.3270954834758197>, <0.05206218099399751,0.30629449152025623,-0.09788595802463446>, 0.5}
    cylinder { m*<-1.5695531788988935,2.634768435938858,-1.0718317234406052>, <0.05206218099399751,0.30629449152025623,-0.09788595802463446>, 0.5 }
    cylinder {  m*<-2.420028000555332,-4.366837572800881,-1.530199408967328>, <0.05206218099399751,0.30629449152025623,-0.09788595802463446>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    