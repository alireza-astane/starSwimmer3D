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
    sphere { m*<-1.131936430153916,-0.17019381839921896,-1.279186807451562>, 1 }        
    sphere {  m*<0.13545013565974245,0.2823543044217074,8.629815872751397>, 1 }
    sphere {  m*<5.730189819035949,0.0706088980334009,-4.757358149878563>, 1 }
    sphere {  m*<-2.7931936931027312,2.1588478085994773,-2.1823921118202376>, 1}
    sphere { m*<-2.5254064720649,-2.72884413380442,-1.9928458266576674>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.13545013565974245,0.2823543044217074,8.629815872751397>, <-1.131936430153916,-0.17019381839921896,-1.279186807451562>, 0.5 }
    cylinder { m*<5.730189819035949,0.0706088980334009,-4.757358149878563>, <-1.131936430153916,-0.17019381839921896,-1.279186807451562>, 0.5}
    cylinder { m*<-2.7931936931027312,2.1588478085994773,-2.1823921118202376>, <-1.131936430153916,-0.17019381839921896,-1.279186807451562>, 0.5 }
    cylinder {  m*<-2.5254064720649,-2.72884413380442,-1.9928458266576674>, <-1.131936430153916,-0.17019381839921896,-1.279186807451562>, 0.5}

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
    sphere { m*<-1.131936430153916,-0.17019381839921896,-1.279186807451562>, 1 }        
    sphere {  m*<0.13545013565974245,0.2823543044217074,8.629815872751397>, 1 }
    sphere {  m*<5.730189819035949,0.0706088980334009,-4.757358149878563>, 1 }
    sphere {  m*<-2.7931936931027312,2.1588478085994773,-2.1823921118202376>, 1}
    sphere { m*<-2.5254064720649,-2.72884413380442,-1.9928458266576674>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.13545013565974245,0.2823543044217074,8.629815872751397>, <-1.131936430153916,-0.17019381839921896,-1.279186807451562>, 0.5 }
    cylinder { m*<5.730189819035949,0.0706088980334009,-4.757358149878563>, <-1.131936430153916,-0.17019381839921896,-1.279186807451562>, 0.5}
    cylinder { m*<-2.7931936931027312,2.1588478085994773,-2.1823921118202376>, <-1.131936430153916,-0.17019381839921896,-1.279186807451562>, 0.5 }
    cylinder {  m*<-2.5254064720649,-2.72884413380442,-1.9928458266576674>, <-1.131936430153916,-0.17019381839921896,-1.279186807451562>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    