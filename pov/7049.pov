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
    sphere { m*<-0.7837140347254412,-1.2273163736570691,-0.6124922176268283>, 1 }        
    sphere {  m*<0.6354534594747218,-0.23737745977715097,9.236797879408329>, 1 }
    sphere {  m*<8.00324065779753,-0.5224697105694129,-5.333879549665613>, 1 }
    sphere {  m*<-6.892722535891472,6.000611663051242,-3.843072646484006>, 1}
    sphere { m*<-2.156247119387072,-4.2164254342384915,-1.2480952002334589>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6354534594747218,-0.23737745977715097,9.236797879408329>, <-0.7837140347254412,-1.2273163736570691,-0.6124922176268283>, 0.5 }
    cylinder { m*<8.00324065779753,-0.5224697105694129,-5.333879549665613>, <-0.7837140347254412,-1.2273163736570691,-0.6124922176268283>, 0.5}
    cylinder { m*<-6.892722535891472,6.000611663051242,-3.843072646484006>, <-0.7837140347254412,-1.2273163736570691,-0.6124922176268283>, 0.5 }
    cylinder {  m*<-2.156247119387072,-4.2164254342384915,-1.2480952002334589>, <-0.7837140347254412,-1.2273163736570691,-0.6124922176268283>, 0.5}

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
    sphere { m*<-0.7837140347254412,-1.2273163736570691,-0.6124922176268283>, 1 }        
    sphere {  m*<0.6354534594747218,-0.23737745977715097,9.236797879408329>, 1 }
    sphere {  m*<8.00324065779753,-0.5224697105694129,-5.333879549665613>, 1 }
    sphere {  m*<-6.892722535891472,6.000611663051242,-3.843072646484006>, 1}
    sphere { m*<-2.156247119387072,-4.2164254342384915,-1.2480952002334589>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6354534594747218,-0.23737745977715097,9.236797879408329>, <-0.7837140347254412,-1.2273163736570691,-0.6124922176268283>, 0.5 }
    cylinder { m*<8.00324065779753,-0.5224697105694129,-5.333879549665613>, <-0.7837140347254412,-1.2273163736570691,-0.6124922176268283>, 0.5}
    cylinder { m*<-6.892722535891472,6.000611663051242,-3.843072646484006>, <-0.7837140347254412,-1.2273163736570691,-0.6124922176268283>, 0.5 }
    cylinder {  m*<-2.156247119387072,-4.2164254342384915,-1.2480952002334589>, <-0.7837140347254412,-1.2273163736570691,-0.6124922176268283>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    