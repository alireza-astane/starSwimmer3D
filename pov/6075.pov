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
    sphere { m*<-1.5459079614249405,-0.2653716752725687,-1.0023158785794082>, 1 }        
    sphere {  m*<-0.08211820069099152,0.20644946978867268,8.878800173776888>, 1 }
    sphere {  m*<7.273233237308982,0.11752919379431531,-5.700693116268473>, 1 }
    sphere {  m*<-3.5456157897655123,2.459433619312772,-2.0278630523398133>, 1}
    sphere { m*<-2.9322480636447112,-2.8361452614621543,-1.6860446987344506>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.08211820069099152,0.20644946978867268,8.878800173776888>, <-1.5459079614249405,-0.2653716752725687,-1.0023158785794082>, 0.5 }
    cylinder { m*<7.273233237308982,0.11752919379431531,-5.700693116268473>, <-1.5459079614249405,-0.2653716752725687,-1.0023158785794082>, 0.5}
    cylinder { m*<-3.5456157897655123,2.459433619312772,-2.0278630523398133>, <-1.5459079614249405,-0.2653716752725687,-1.0023158785794082>, 0.5 }
    cylinder {  m*<-2.9322480636447112,-2.8361452614621543,-1.6860446987344506>, <-1.5459079614249405,-0.2653716752725687,-1.0023158785794082>, 0.5}

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
    sphere { m*<-1.5459079614249405,-0.2653716752725687,-1.0023158785794082>, 1 }        
    sphere {  m*<-0.08211820069099152,0.20644946978867268,8.878800173776888>, 1 }
    sphere {  m*<7.273233237308982,0.11752919379431531,-5.700693116268473>, 1 }
    sphere {  m*<-3.5456157897655123,2.459433619312772,-2.0278630523398133>, 1}
    sphere { m*<-2.9322480636447112,-2.8361452614621543,-1.6860446987344506>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.08211820069099152,0.20644946978867268,8.878800173776888>, <-1.5459079614249405,-0.2653716752725687,-1.0023158785794082>, 0.5 }
    cylinder { m*<7.273233237308982,0.11752919379431531,-5.700693116268473>, <-1.5459079614249405,-0.2653716752725687,-1.0023158785794082>, 0.5}
    cylinder { m*<-3.5456157897655123,2.459433619312772,-2.0278630523398133>, <-1.5459079614249405,-0.2653716752725687,-1.0023158785794082>, 0.5 }
    cylinder {  m*<-2.9322480636447112,-2.8361452614621543,-1.6860446987344506>, <-1.5459079614249405,-0.2653716752725687,-1.0023158785794082>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    