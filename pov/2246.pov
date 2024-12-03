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
    sphere { m*<1.0883090343797088,0.3279857111371594,0.5093480316836554>, 1 }        
    sphere {  m*<1.332393962081721,0.3535748399853228,3.49929137663893>, 1 }
    sphere {  m*<3.8256411511442554,0.35357483998532274,-0.717990831851687>, 1 }
    sphere {  m*<-3.107703089561715,7.009883145715218,-1.971607464369365>, 1}
    sphere { m*<-3.776665601626815,-7.921380423738634,-2.366463828430729>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.332393962081721,0.3535748399853228,3.49929137663893>, <1.0883090343797088,0.3279857111371594,0.5093480316836554>, 0.5 }
    cylinder { m*<3.8256411511442554,0.35357483998532274,-0.717990831851687>, <1.0883090343797088,0.3279857111371594,0.5093480316836554>, 0.5}
    cylinder { m*<-3.107703089561715,7.009883145715218,-1.971607464369365>, <1.0883090343797088,0.3279857111371594,0.5093480316836554>, 0.5 }
    cylinder {  m*<-3.776665601626815,-7.921380423738634,-2.366463828430729>, <1.0883090343797088,0.3279857111371594,0.5093480316836554>, 0.5}

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
    sphere { m*<1.0883090343797088,0.3279857111371594,0.5093480316836554>, 1 }        
    sphere {  m*<1.332393962081721,0.3535748399853228,3.49929137663893>, 1 }
    sphere {  m*<3.8256411511442554,0.35357483998532274,-0.717990831851687>, 1 }
    sphere {  m*<-3.107703089561715,7.009883145715218,-1.971607464369365>, 1}
    sphere { m*<-3.776665601626815,-7.921380423738634,-2.366463828430729>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.332393962081721,0.3535748399853228,3.49929137663893>, <1.0883090343797088,0.3279857111371594,0.5093480316836554>, 0.5 }
    cylinder { m*<3.8256411511442554,0.35357483998532274,-0.717990831851687>, <1.0883090343797088,0.3279857111371594,0.5093480316836554>, 0.5}
    cylinder { m*<-3.107703089561715,7.009883145715218,-1.971607464369365>, <1.0883090343797088,0.3279857111371594,0.5093480316836554>, 0.5 }
    cylinder {  m*<-3.776665601626815,-7.921380423738634,-2.366463828430729>, <1.0883090343797088,0.3279857111371594,0.5093480316836554>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    