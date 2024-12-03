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
    sphere { m*<-0.20093919852584058,-0.10265082172495435,-0.8493831490892727>, 1 }        
    sphere {  m*<0.31166608450413447,0.1714158380803514,5.512116810216752>, 1 }
    sphere {  m*<2.5337691954804167,-0.0006168463385801404,-2.0785926745404537>, 1 }
    sphere {  m*<-1.8225545584187306,2.2258231226936447,-1.8233289145052407>, 1}
    sphere { m*<-1.5547673373808988,-2.6618688197102527,-1.6337826293426678>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.31166608450413447,0.1714158380803514,5.512116810216752>, <-0.20093919852584058,-0.10265082172495435,-0.8493831490892727>, 0.5 }
    cylinder { m*<2.5337691954804167,-0.0006168463385801404,-2.0785926745404537>, <-0.20093919852584058,-0.10265082172495435,-0.8493831490892727>, 0.5}
    cylinder { m*<-1.8225545584187306,2.2258231226936447,-1.8233289145052407>, <-0.20093919852584058,-0.10265082172495435,-0.8493831490892727>, 0.5 }
    cylinder {  m*<-1.5547673373808988,-2.6618688197102527,-1.6337826293426678>, <-0.20093919852584058,-0.10265082172495435,-0.8493831490892727>, 0.5}

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
    sphere { m*<-0.20093919852584058,-0.10265082172495435,-0.8493831490892727>, 1 }        
    sphere {  m*<0.31166608450413447,0.1714158380803514,5.512116810216752>, 1 }
    sphere {  m*<2.5337691954804167,-0.0006168463385801404,-2.0785926745404537>, 1 }
    sphere {  m*<-1.8225545584187306,2.2258231226936447,-1.8233289145052407>, 1}
    sphere { m*<-1.5547673373808988,-2.6618688197102527,-1.6337826293426678>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.31166608450413447,0.1714158380803514,5.512116810216752>, <-0.20093919852584058,-0.10265082172495435,-0.8493831490892727>, 0.5 }
    cylinder { m*<2.5337691954804167,-0.0006168463385801404,-2.0785926745404537>, <-0.20093919852584058,-0.10265082172495435,-0.8493831490892727>, 0.5}
    cylinder { m*<-1.8225545584187306,2.2258231226936447,-1.8233289145052407>, <-0.20093919852584058,-0.10265082172495435,-0.8493831490892727>, 0.5 }
    cylinder {  m*<-1.5547673373808988,-2.6618688197102527,-1.6337826293426678>, <-0.20093919852584058,-0.10265082172495435,-0.8493831490892727>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    