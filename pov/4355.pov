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
    sphere { m*<-0.18590404452599735,-0.0946122107843128,-0.6627948818111735>, 1 }        
    sphere {  m*<0.2548016872211258,0.14101303900446913,4.806422052520179>, 1 }
    sphere {  m*<2.54880434948026,0.007421764602061384,-1.8920044072623556>, 1 }
    sphere {  m*<-1.8075194044188874,2.233861733634286,-1.6367406472271424>, 1}
    sphere { m*<-1.5397321833810556,-2.6538302087696115,-1.4471943620645698>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2548016872211258,0.14101303900446913,4.806422052520179>, <-0.18590404452599735,-0.0946122107843128,-0.6627948818111735>, 0.5 }
    cylinder { m*<2.54880434948026,0.007421764602061384,-1.8920044072623556>, <-0.18590404452599735,-0.0946122107843128,-0.6627948818111735>, 0.5}
    cylinder { m*<-1.8075194044188874,2.233861733634286,-1.6367406472271424>, <-0.18590404452599735,-0.0946122107843128,-0.6627948818111735>, 0.5 }
    cylinder {  m*<-1.5397321833810556,-2.6538302087696115,-1.4471943620645698>, <-0.18590404452599735,-0.0946122107843128,-0.6627948818111735>, 0.5}

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
    sphere { m*<-0.18590404452599735,-0.0946122107843128,-0.6627948818111735>, 1 }        
    sphere {  m*<0.2548016872211258,0.14101303900446913,4.806422052520179>, 1 }
    sphere {  m*<2.54880434948026,0.007421764602061384,-1.8920044072623556>, 1 }
    sphere {  m*<-1.8075194044188874,2.233861733634286,-1.6367406472271424>, 1}
    sphere { m*<-1.5397321833810556,-2.6538302087696115,-1.4471943620645698>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2548016872211258,0.14101303900446913,4.806422052520179>, <-0.18590404452599735,-0.0946122107843128,-0.6627948818111735>, 0.5 }
    cylinder { m*<2.54880434948026,0.007421764602061384,-1.8920044072623556>, <-0.18590404452599735,-0.0946122107843128,-0.6627948818111735>, 0.5}
    cylinder { m*<-1.8075194044188874,2.233861733634286,-1.6367406472271424>, <-0.18590404452599735,-0.0946122107843128,-0.6627948818111735>, 0.5 }
    cylinder {  m*<-1.5397321833810556,-2.6538302087696115,-1.4471943620645698>, <-0.18590404452599735,-0.0946122107843128,-0.6627948818111735>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    